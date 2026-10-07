from datetime import datetime, timedelta, timezone
from itertools import count

import pytest
from sqlalchemy import func, select

from matcher import match_pending
from matcher.location import location_similarity, normalize_location
from models import Classification, Incident, IncidentItem, RawItem, Source, hash_url

T0 = datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)
NOTHING = {
    "created": 0, "merged": 0, "skipped_rejected": 0, "skipped_no_time": 0,
    "skipped_followup_no_time": 0, "skipped_followup_no_match": 0, "refreshed": 0,
}
_ids = count(1)
MIDNIGHT = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)  # 00:00 CDT


@pytest.fixture
def add(session):
    source = Source(name="KSDK", url="https://example.com/feed", type="rss")
    session.add(source)
    session.flush()

    def _add(
        *,
        crime_type="shooting",
        occurred_at=T0,
        location="1200 block of N Grand Blvd",
        confidence=0.9,
        is_crime=True,
        in_stl=True,
        published_at=None,
        was_shooting=False,
        neighborhood=None,
        is_followup=False,
        time_precision="exact",
    ) -> RawItem:
        n = next(_ids)
        url = f"https://example.com/story/{n}"
        item = RawItem(
            source_id=source.id,
            url=url,
            url_hash=hash_url(url),
            title=f"story {n}",
            # Published an hour after the crime unless a test says otherwise.
            published_at=published_at or (occurred_at or T0) + timedelta(hours=1),
            status="classified",
        )
        session.add(item)
        session.flush()
        session.add(
            Classification(
                raw_item_id=item.id,
                is_crime=is_crime,
                crime_type=crime_type if is_crime else None,
                was_shooting=was_shooting,
                in_stl=in_stl,
                occurred_at=occurred_at,
                time_precision=time_precision,
                location=location,
                neighborhood=neighborhood,
                is_followup=is_followup,
                confidence=confidence,
                model="test",
                prompt_version="t",
            )
        )
        session.commit()
        return item

    return _add


def incidents(session) -> list[Incident]:
    return session.scalars(select(Incident).order_by(Incident.id)).all()


def links(session) -> int:
    return session.scalar(select(func.count()).select_from(IncidentItem))


def test_normalize_location():
    assert normalize_location("1200 Block of N. Grand Blvd, St. Louis, MO") == (
        "1200 block of n grand blvd mo"
    )
    assert normalize_location("Saint Louis") == ""
    assert normalize_location("City of St Louis") == ""
    assert normalize_location(None) == ""


def test_location_similarity():
    assert location_similarity("N. Grand Blvd, St. Louis", "n grand blvd") == 100
    assert location_similarity("Grand Blvd", "Delmar Blvd and Skinker") < 85
    assert location_similarity(None, "Grand Blvd") == 0
    assert location_similarity("St. Louis", "St. Louis") == 0


def test_new_incident(session, add):
    item = add()
    assert match_pending(session, threshold=85) == {**NOTHING, "created": 1}
    [inc] = incidents(session)
    assert inc.crime_type == "shooting"
    assert inc.occurred_at == T0
    assert inc.status == "confirmed"
    assert inc.time_estimated is False
    assert [r.id for r in inc.raw_items] == [item.id]


def test_null_occurred_at_falls_back_to_published_at(session, add):
    published = T0 + timedelta(hours=2)
    add(occurred_at=None, published_at=published)
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.occurred_at == published
    assert inc.time_estimated is True


def test_reported_time_replaces_estimated(session, add):
    add(occurred_at=None, published_at=T0 + timedelta(hours=3))
    add(occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.occurred_at == T0 + timedelta(hours=1)
    assert inc.time_estimated is False


def test_merge_from_two_sources_keeps_earliest(session, add):
    add(occurred_at=T0 + timedelta(hours=2), location="1200 block of North Grand")
    add(occurred_at=T0, location="1200 Block of N. Grand Blvd., St. Louis")
    counts = match_pending(session, threshold=80)
    assert counts == {**NOTHING, "created": 1, "merged": 1}
    [inc] = incidents(session)
    assert inc.occurred_at == T0
    assert len(inc.raw_items) == 2


def test_shooting_then_homicide_upgrades(session, add):
    add(crime_type="shooting")
    add(crime_type="homicide", occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.crime_type == "homicide"
    assert inc.was_shooting is True
    assert len(inc.raw_items) == 2


def test_homicide_then_shooting_stays_homicide(session, add):
    add(crime_type="homicide")
    match_pending(session, threshold=85)
    assert incidents(session)[0].was_shooting is False
    add(crime_type="shooting", occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.crime_type == "homicide"
    assert inc.was_shooting is True
    assert len(inc.raw_items) == 2


@pytest.mark.parametrize(
    "crime_type, was_shooting, expected",
    [
        ("shooting", False, True),  # a v1/v2 classification without the flag
        ("shooting", True, True),
        ("homicide", True, True),
        ("homicide", False, False),
        ("burglary", False, False),
    ],
)
def test_new_incident_was_shooting(session, add, crime_type, was_shooting, expected):
    add(crime_type=crime_type, was_shooting=was_shooting)
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.was_shooting is expected


def test_was_shooting_kept_when_unshot_homicide_merges(session, add):
    add(crime_type="homicide", was_shooting=True)
    add(crime_type="homicide", occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.was_shooting is True
    assert len(inc.raw_items) == 2


def test_burglary_does_not_merge_with_shooting(session, add):
    add(crime_type="shooting")
    add(crime_type="burglary")
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_outside_window_creates_separate_incidents(session, add):
    add(occurred_at=T0)
    add(occurred_at=T0 + timedelta(hours=7))
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_edge_of_window_merges(session, add):
    add(occurred_at=T0)
    add(occurred_at=T0 + timedelta(hours=6))
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 1


def test_different_locations_create_separate_incidents(session, add):
    add(location="1200 block of N Grand Blvd")
    add(location="Delmar Blvd and Skinker Blvd")
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_missing_location_does_not_merge(session, add):
    add(location=None)
    add(location=None)
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_threshold_from_settings(session, add, monkeypatch):
    from config import get_settings

    monkeypatch.setenv("MATCH_LOCATION_THRESHOLD", "10")
    get_settings.cache_clear()
    try:
        add(location="Grand Blvd")
        add(location="Delmar Ave")
        match_pending(session)
    finally:
        get_settings.cache_clear()
    assert len(incidents(session)) == 1


@pytest.mark.parametrize("kwargs", [{"in_stl": False}, {"is_crime": False}])
def test_non_stl_or_non_crime_ignored(session, add, kwargs):
    add(**kwargs)
    assert match_pending(session, threshold=85) == NOTHING
    assert incidents(session) == []
    assert links(session) == 0


def test_newest_classification_wins(session, add):
    item = add()
    session.add(
        Classification(
            raw_item_id=item.id,
            is_crime=True,
            crime_type="shooting",
            in_stl=False,
            occurred_at=T0,
            location="x",
            confidence=0.9,
            model="test",
            prompt_version="t",
        )
    )
    session.commit()
    match_pending(session, threshold=85)
    assert incidents(session) == []


def test_low_confidence_goes_to_review(session, add):
    add(confidence=0.6)
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.status == "review"


def test_confident_item_confirms_review_incident(session, add):
    add(confidence=0.6)
    match_pending(session, threshold=85)
    add(confidence=0.9, occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.status == "confirmed"


def test_low_confidence_does_not_downgrade_confirmed(session, add):
    add(confidence=0.9)
    add(confidence=0.5, occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.status == "confirmed"


def test_rejected_incident_untouched(session, add):
    rejected = Incident(
        crime_type="shooting",
        occurred_at=T0,
        location="1200 block of N Grand Blvd",
        status="rejected",
    )
    session.add(rejected)
    session.commit()

    add(crime_type="homicide", occurred_at=T0 - timedelta(hours=1), confidence=0.95)
    assert match_pending(session, threshold=85) == {**NOTHING, "skipped_rejected": 1}

    [inc] = incidents(session)
    assert inc.id == rejected.id
    assert inc.status == "rejected"
    assert inc.crime_type == "shooting"
    assert inc.occurred_at == T0
    assert links(session) == 0


def test_live_incident_preferred_over_rejected(session, add):
    session.add(
        Incident(
            crime_type="shooting", occurred_at=T0, location="N Grand Blvd", status="rejected"
        )
    )
    live = Incident(
        crime_type="shooting", occurred_at=T0, location="N Grand Blvd", status="review"
    )
    session.add(live)
    session.commit()
    add(location="N Grand Blvd")
    assert match_pending(session, threshold=85) == {**NOTHING, "merged": 1}
    assert len(live.raw_items) == 1


def test_running_twice_is_idempotent(session, add):
    add()
    add(occurred_at=T0 + timedelta(hours=1))
    add(location="Delmar Blvd and Skinker Blvd")
    first = match_pending(session, threshold=85)
    snapshot = [(i.id, i.crime_type, i.occurred_at, i.status) for i in incidents(session)]

    second = match_pending(session, threshold=85)

    assert first == {**NOTHING, "created": 2, "merged": 1}
    assert second == NOTHING
    assert [
        (i.id, i.crime_type, i.occurred_at, i.status) for i in incidents(session)
    ] == snapshot
    assert links(session) == 3


def reclassify(session, item, **fields):
    """Add a newer classification for an already-classified item."""
    base = {
        "is_crime": True, "crime_type": "shooting", "in_stl": True, "occurred_at": T0,
        "location": "1200 block of N Grand Blvd", "confidence": 0.9,
    }
    session.add(Classification(
        raw_item_id=item.id, model="test", prompt_version="t2", **{**base, **fields}
    ))
    session.commit()


# --- neighborhood matching ---------------------------------------------------

def test_same_neighborhood_merges_despite_different_location(session, add):
    add(location="1200 block of Hamilton Avenue", neighborhood="West End")
    add(location="West End", neighborhood="West End", occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert inc.neighborhood == "West End"


def test_different_neighborhoods_and_locations_stay_separate(session, add):
    add(location="1200 block of Hamilton Avenue", neighborhood="West End")
    add(location="Arsenal Street", neighborhood="Tower Grove South")
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_missing_neighborhood_falls_back_to_location(session, add):
    add(location="West End", neighborhood="West End")
    add(location="1200 block of Hamilton Avenue", neighborhood=None)
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2  # no neighborhood to compare, locations differ


def test_neighborhood_match_still_needs_time_window(session, add):
    add(neighborhood="Dutchtown", location="Meramec St")
    add(neighborhood="Dutchtown", location="Gravois Ave", occurred_at=T0 + timedelta(hours=7))
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_merged_incident_fills_neighborhood(session, add):
    add(neighborhood=None)
    add(neighborhood="Fox Park", occurred_at=T0 + timedelta(hours=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert inc.neighborhood == "Fox Park"


# --- wider window for estimated times ----------------------------------------

def test_estimated_incident_matches_within_24h(session, add):
    add(occurred_at=None, published_at=T0)  # estimated incident at T0
    match_pending(session, threshold=85)
    add(occurred_at=T0 - timedelta(hours=20))  # reported time, 20h earlier
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert inc.time_estimated is False
    assert inc.occurred_at == T0 - timedelta(hours=20)


def test_estimated_item_matches_reported_incident_within_24h(session, add):
    add(occurred_at=T0)
    match_pending(session, threshold=85)
    add(occurred_at=None, published_at=T0 + timedelta(hours=20))
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 1


def test_estimated_window_ends_at_24h(session, add):
    add(occurred_at=None, published_at=T0)
    add(occurred_at=T0 + timedelta(hours=25))
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


def test_reported_times_keep_6h_window(session, add):
    add(occurred_at=T0)
    add(occurred_at=T0 + timedelta(hours=20))
    match_pending(session, threshold=85)
    assert len(incidents(session)) == 2


# --- follow-ups --------------------------------------------------------------

def test_followup_without_time_is_skipped(session, add):
    add(occurred_at=None, published_at=T0 + timedelta(days=3), is_followup=True)
    counts = match_pending(session, threshold=85)
    assert counts == {**NOTHING, "skipped_followup_no_time": 1}
    assert incidents(session) == []


def test_followup_with_time_merges_into_original(session, add):
    add(crime_type="homicide", was_shooting=True)
    add(
        crime_type="homicide", occurred_at=T0 + timedelta(minutes=30),
        published_at=T0 + timedelta(days=2), is_followup=True,
    )
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert inc.occurred_at == T0  # the follow-up's publish date never moves it


def test_followup_without_match_creates_no_incident(session, add):
    # A confident follow-up with a stated time is still not a new crime.
    add(is_followup=True, confidence=0.95, published_at=T0 + timedelta(days=2))
    counts = match_pending(session, threshold=85)
    assert counts == {**NOTHING, "skipped_followup_no_match": 1}
    assert incidents(session) == []
    assert links(session) == 0


def test_date_only_detention_story_creates_no_incident(session, add):
    # Fox 2, 2026-10-06: "Police detain person of interest in connection to
    # burglaries" was classified with the detention night as occurred_at.
    add(crime_type="burglary", occurred_at=datetime(2026, 10, 6, 7, tzinfo=timezone.utc),
        time_precision="date_only", location="9th St. and Allen Avenue",
        neighborhood="Soulard", is_followup=True, confidence=0.95,
        published_at=datetime(2026, 10, 6, 11, 13, tzinfo=timezone.utc))
    counts = match_pending(session, threshold=85)
    assert counts["created"] == 0
    assert counts["skipped_followup_no_match"] == 1
    assert incidents(session) == []


# Eval case 25 as gemini-3.5-flash-lite classified it under prompt v6: the
# detention day (Tue 00:00 CDT) guessed as the crime time.
DETENTION = dict(
    crime_type="burglary", occurred_at=datetime(2026, 10, 6, 5, tzinfo=timezone.utc),
    time_precision="date_only", location="9th St. and Allen Avenue", neighborhood="Soulard",
    is_followup=True, confidence=0.9,
    published_at=datetime(2026, 10, 6, 11, 13, tzinfo=timezone.utc),
)
EARLY_TUESDAY = datetime(2026, 10, 6, 7, 30, tzinfo=timezone.utc)  # 02:30 CDT


@pytest.mark.parametrize("precision", ["date_only", "exact"])
def test_detention_story_never_moves_incident_time(session, add, precision):
    # A real Soulard burglary, later than the detention story's guessed midnight.
    add(crime_type="burglary", occurred_at=EARLY_TUESDAY, time_precision=precision,
        location="Allen Avenue", neighborhood="Soulard", confidence=0.6,
        published_at=EARLY_TUESDAY + timedelta(hours=3))
    match_pending(session, threshold=85)
    add(**DETENTION)
    counts = match_pending(session, threshold=85)
    assert counts["merged"] == 1  # it still enriches the incident...
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert inc.status == "confirmed"
    # ...but its guessed time is ignored, by the merge and by the refresh.
    assert inc.occurred_at == EARLY_TUESDAY
    assert inc.time_estimated is (precision != "exact")
    match_pending(session, threshold=85)
    assert incidents(session)[0].occurred_at == EARLY_TUESDAY


def test_exact_followup_time_still_replaces_estimated(session, add):
    add(occurred_at=MIDNIGHT, time_precision="date_only")
    add(occurred_at=MIDNIGHT + timedelta(hours=2), time_precision="exact",
        is_followup=True, published_at=MIDNIGHT + timedelta(days=1))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (MIDNIGHT + timedelta(hours=2), False)


def test_followup_before_original_links_on_later_run(session, add):
    add(is_followup=True, occurred_at=T0 + timedelta(minutes=10),
        published_at=T0 + timedelta(days=1))
    assert match_pending(session, threshold=85) == {**NOTHING, "skipped_followup_no_match": 1}

    add()  # the original report arrives later
    counts = match_pending(session, threshold=85)
    assert (counts["created"], counts["merged"], counts["skipped_followup_no_match"]) == (1, 1, 0)
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert inc.occurred_at == T0


def test_followup_enriches_existing_incident(session, add):
    add(crime_type="shooting", location=None, neighborhood="Soulard", confidence=0.5)
    match_pending(session, threshold=85)
    add(crime_type="homicide", was_shooting=True, neighborhood="Soulard",
        location="9th St. and Allen Avenue", is_followup=True, confidence=0.9,
        occurred_at=T0 + timedelta(minutes=20), published_at=T0 + timedelta(days=1))
    counts = match_pending(session, threshold=85)
    assert counts == {**NOTHING, "merged": 1}
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert (inc.crime_type, inc.was_shooting, inc.location, inc.status) == (
        "homicide", True, "9th St. and Allen Avenue", "confirmed"
    )
    assert inc.occurred_at == T0


def test_followup_matching_rejected_incident_is_skipped(session, add):
    add()
    match_pending(session, threshold=85)
    incidents(session)[0].status = "rejected"
    session.commit()
    add(is_followup=True, occurred_at=T0 + timedelta(minutes=5),
        published_at=T0 + timedelta(days=1))
    assert match_pending(session, threshold=85) == {**NOTHING, "skipped_rejected": 1}


def test_first_report_still_creates_incident(session, add):
    add(is_followup=False)
    assert match_pending(session, threshold=85)["created"] == 1


# --- refresh from re-classified linked items ---------------------------------

def test_refresh_takes_reported_time_from_reclassified_item(session, add):
    item = add(occurred_at=T0, time_precision="date_only", published_at=T0 + timedelta(hours=20))
    match_pending(session, threshold=85)
    assert incidents(session)[0].time_estimated is True

    reclassify(session, item, occurred_at=T0 + timedelta(hours=12))
    counts = match_pending(session, threshold=85)

    [inc] = incidents(session)
    assert counts["refreshed"] == 1
    # A reported time replaces the estimate, even though it's later.
    assert (inc.occurred_at, inc.time_estimated) == (T0 + timedelta(hours=12), False)
    assert match_pending(session, threshold=85)["refreshed"] == 0  # idempotent


def test_refresh_uses_earliest_exact_time_across_items(session, add):
    a = add(occurred_at=T0)
    add(occurred_at=T0 + timedelta(hours=2))
    match_pending(session, threshold=85)

    reclassify(session, a, occurred_at=T0 - timedelta(hours=1))
    match_pending(session, threshold=85)
    assert incidents(session)[0].occurred_at == T0 - timedelta(hours=1)

    # Corrected to later than the other item: the other item's time is now earliest.
    reclassify(session, a, occurred_at=T0 + timedelta(hours=5))
    match_pending(session, threshold=85)
    assert incidents(session)[0].occurred_at == T0 + timedelta(hours=2)




def test_exact_time_beats_guessed_midnight_on_merge(session, add):
    add(occurred_at=MIDNIGHT, time_precision="date_only")
    add(occurred_at=MIDNIGHT + timedelta(minutes=40))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert len(inc.raw_items) == 2
    assert (inc.occurred_at, inc.time_estimated) == (MIDNIGHT + timedelta(minutes=40), False)


def test_date_only_incident_is_estimated_and_uses_24h_window(session, add):
    add(occurred_at=MIDNIGHT, time_precision="date_only")
    match_pending(session, threshold=85)
    assert incidents(session)[0].time_estimated is True
    add(occurred_at=MIDNIGHT + timedelta(hours=20))  # exact, 20h later
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (MIDNIGHT + timedelta(hours=20), False)


def test_refresh_replaces_guessed_midnight_with_later_exact_time(session, add):
    a = add(occurred_at=MIDNIGHT)  # stored as exact, like the old midnight guesses
    add(occurred_at=MIDNIGHT + timedelta(minutes=40))
    match_pending(session, threshold=85)
    assert incidents(session)[0].occurred_at == MIDNIGHT

    reclassify(session, a, occurred_at=MIDNIGHT, time_precision="date_only")
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (MIDNIGHT + timedelta(minutes=40), False)


def test_refresh_leaves_time_alone_without_exact_times(session, add):
    a = add(occurred_at=MIDNIGHT)
    match_pending(session, threshold=85)
    reclassify(session, a, occurred_at=MIDNIGHT + timedelta(hours=3), time_precision="date_only")
    assert match_pending(session, threshold=85)["refreshed"] == 0
    assert incidents(session)[0].occurred_at == MIDNIGHT


def test_date_only_followup_is_linked_with_estimated_time(session, add):
    add(occurred_at=MIDNIGHT, time_precision="date_only")
    add(occurred_at=MIDNIGHT + timedelta(hours=1), time_precision="date_only", is_followup=True,
        published_at=MIDNIGHT + timedelta(days=3))
    counts = match_pending(session, threshold=85)
    assert (counts["created"], counts["merged"]) == (1, 1)
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (MIDNIGHT, True)


def test_refresh_turns_on_was_shooting_never_off(session, add):
    item = add(crime_type="homicide")
    match_pending(session, threshold=85)
    reclassify(session, item, crime_type="homicide", was_shooting=True)
    match_pending(session, threshold=85)
    assert incidents(session)[0].was_shooting is True

    reclassify(session, item, crime_type="homicide", was_shooting=False)
    match_pending(session, threshold=85)
    assert incidents(session)[0].was_shooting is True


def test_refresh_ignores_items_now_not_crime(session, add):
    item = add(occurred_at=None, published_at=T0)
    match_pending(session, threshold=85)
    reclassify(session, item, is_crime=False, crime_type=None, occurred_at=T0 - timedelta(hours=3))
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (T0, True)


@pytest.mark.parametrize("status", ["rejected", "merged"])
def test_refresh_leaves_rejected_and_merged_alone(session, add, status):
    item = add(occurred_at=None, published_at=T0)
    match_pending(session, threshold=85)
    inc = incidents(session)[0]
    inc.status = status
    session.commit()

    reclassify(session, item, occurred_at=T0 - timedelta(hours=1), was_shooting=True)
    assert match_pending(session, threshold=85)["refreshed"] == 0
    session.refresh(inc)
    assert (inc.occurred_at, inc.time_estimated) == (T0, True)


def test_refresh_fills_neighborhood(session, add):
    item = add()
    match_pending(session, threshold=85)
    reclassify(session, item, neighborhood="Shaw")
    match_pending(session, threshold=85)
    assert incidents(session)[0].neighborhood == "Shaw"


# --- merged incidents --------------------------------------------------------

def test_merged_incident_is_not_a_match_candidate(session, add):
    add()
    match_pending(session, threshold=85)
    merged = incidents(session)[0]
    merged.status = "merged"
    session.commit()

    add(occurred_at=T0 + timedelta(hours=1))
    counts = match_pending(session, threshold=85)

    assert counts["created"] == 1
    assert len(merged.raw_items) == 1


# --- publish-time cap ----------------------------------------------------------

def test_occurred_after_publish_is_capped_and_estimated(session, add):
    # The KSDK burglary string: model said Oct 5 00:00 for a story published earlier
    # than a later guess would allow.
    published = T0 + timedelta(hours=1)
    add(occurred_at=T0 + timedelta(hours=5), published_at=published)
    match_pending(session, threshold=85)
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (published, True)


def test_capped_time_never_counts_as_reported_on_refresh(session, add):
    item = add(occurred_at=T0, time_precision="date_only", published_at=T0 + timedelta(hours=3))
    match_pending(session, threshold=85)

    reclassify(session, item, occurred_at=T0 + timedelta(hours=8))  # "exact", after publish
    assert match_pending(session, threshold=85)["refreshed"] == 0
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (T0, True)


# --- manual (admin-edited) fields ---------------------------------------------

def _make_manual(session, **fields):
    [inc] = incidents(session)
    for name, value in fields.items():
        setattr(inc, name, value)
    session.commit()
    return inc


def test_refresh_never_overwrites_manual_time(session, add):
    item = add(occurred_at=T0, time_precision="date_only")
    match_pending(session, threshold=85)
    manual = T0 - timedelta(hours=20)
    _make_manual(session, occurred_at=manual, time_estimated=True, manual_occurred_at=True)

    reclassify(session, item, occurred_at=T0 - timedelta(minutes=30))  # exact
    assert match_pending(session, threshold=85)["refreshed"] == 0
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (manual, True)


def test_merge_never_moves_manual_time(session, add):
    add(occurred_at=T0, time_precision="date_only")
    match_pending(session, threshold=85)
    manual = T0 + timedelta(hours=2)
    _make_manual(session, occurred_at=manual, time_estimated=True, manual_occurred_at=True)

    add(occurred_at=T0 + timedelta(hours=1))  # exact, earlier: would normally win
    counts = match_pending(session, threshold=85)
    assert counts["merged"] == 1
    [inc] = incidents(session)
    assert (inc.occurred_at, inc.time_estimated) == (manual, True)


def test_manual_cleared_location_stays_cleared(session, add):
    item = add(location="Wrong St", neighborhood="Shaw")
    match_pending(session, threshold=85)
    _make_manual(session, location=None, manual_location=True)

    reclassify(session, item, location="Other St", neighborhood="Shaw")
    match_pending(session, threshold=85)
    # Merges by neighborhood; its location must not fill the cleared one.
    add(occurred_at=T0 + timedelta(minutes=10), location="Third St", neighborhood="Shaw")
    assert match_pending(session, threshold=85)["merged"] == 1
    [inc] = incidents(session)
    assert (inc.location, inc.neighborhood) == (None, "Shaw")


def test_manual_cleared_neighborhood_stays_cleared(session, add):
    item = add(neighborhood="Soulard")
    match_pending(session, threshold=85)
    _make_manual(session, neighborhood=None, manual_neighborhood=True)

    reclassify(session, item, neighborhood="Shaw")
    match_pending(session, threshold=85)
    # Merges by location; its neighborhood must not fill the cleared one.
    add(occurred_at=T0 + timedelta(minutes=10), neighborhood="Shaw")
    assert match_pending(session, threshold=85)["merged"] == 1
    [inc] = incidents(session)
    assert inc.neighborhood is None
