from datetime import datetime, timedelta, timezone
from itertools import count

import pytest
from sqlalchemy import func, select

from matcher import match_pending
from matcher.location import location_similarity, normalize_location
from models import Classification, Incident, IncidentItem, RawItem, Source, hash_url

T0 = datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)
NOTHING = {"created": 0, "merged": 0, "skipped_rejected": 0, "skipped_no_time": 0}
_ids = count(1)


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
    ) -> RawItem:
        n = next(_ids)
        url = f"https://example.com/story/{n}"
        item = RawItem(
            source_id=source.id,
            url=url,
            url_hash=hash_url(url),
            title=f"story {n}",
            published_at=published_at or T0 + timedelta(hours=1),
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
                location=location,
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
