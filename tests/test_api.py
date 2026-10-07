from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.main import app, get_admin_token, get_now, get_session
from models import Incident, IncidentItem, RawItem, Source, hash_url

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
TOKEN = "s3cret"
AUTH = {"X-Admin-Token": TOKEN}


@pytest.fixture
def client(session_factory):
    def override():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = override
    app.dependency_overrides[get_now] = lambda: NOW
    app.dependency_overrides[get_admin_token] = lambda: TOKEN
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def add(session_factory):
    def _add(
        crime_type="shooting",
        hours_ago=1.0,
        status="confirmed",
        location="Grand Blvd",
        articles=0,
        was_shooting=None,
        time_estimated=False,
        neighborhood=None,
    ):
        with session_factory() as s:
            incident = Incident(
                crime_type=crime_type,
                occurred_at=NOW - timedelta(hours=hours_ago),
                location=location,
                neighborhood=neighborhood,
                time_estimated=time_estimated,
                status=status,
                # Like the matcher: a shooting always has was_shooting.
                was_shooting=crime_type == "shooting" if was_shooting is None else was_shooting,
            )
            s.add(incident)
            s.flush()
            if articles:
                source = Source(name="KSDK", url=f"https://ksdk.example/{incident.id}", type="rss")
                s.add(source)
                s.flush()
                for n in range(articles):
                    url = f"https://ksdk.example/story/{incident.id}/{n}"
                    item = RawItem(
                        source_id=source.id,
                        url=url,
                        url_hash=hash_url(url),
                        title=f"Story {n}",
                        published_at=NOW - timedelta(hours=hours_ago - n - 0.5),
                    )
                    s.add(item)
                    s.flush()
                    s.add(IncidentItem(incident_id=incident.id, raw_item_id=item.id))
            s.commit()
            return incident.id

    return _add


def by_type(body) -> dict:
    return {e["crime_type"]: e for e in body["by_type"]}


# --- /timer ---


def test_timer_no_incidents(client):
    body = client.get("/timer").json()
    assert body["now"] == "2026-10-05T12:00:00Z"
    assert body["overall"] == {"crime_type": None, "last": None}
    assert body["by_type"] == [
        {"crime_type": "shooting", "last": None},
        {"crime_type": "burglary", "last": None},
        {"crime_type": "homicide", "last": None},
    ]


def test_timer_only_confirmed_counts(client, add):
    shooting = add("shooting", hours_ago=10)
    burglary = add("burglary", hours_ago=3, location="Delmar")
    add("shooting", hours_ago=1, status="review")
    add("homicide", hours_ago=0.5, status="rejected")

    body = client.get("/timer").json()
    assert body["overall"]["last"] == {
        "incident_id": burglary,
        "crime_type": "burglary",
        "occurred_at": "2026-10-05T09:00:00Z",
        "seconds_since": 3 * 3600,
        "location": "Delmar",
        "time_estimated": False,
        "was_shooting": False,
    }
    types = by_type(body)
    assert types["shooting"]["last"]["incident_id"] == shooting
    assert types["shooting"]["last"]["seconds_since"] == 10 * 3600
    assert types["burglary"]["last"]["incident_id"] == burglary
    assert types["homicide"]["last"] is None


def test_fatal_shooting_counts_for_shooting_and_homicide(client, add):
    add("shooting", hours_ago=20)
    fatal = add("homicide", hours_ago=5, was_shooting=True)

    types = by_type(client.get("/timer").json())
    assert types["shooting"]["last"]["incident_id"] == fatal
    assert types["shooting"]["last"]["crime_type"] == "homicide"
    assert types["shooting"]["last"]["was_shooting"] is True
    assert types["shooting"]["last"]["seconds_since"] == 5 * 3600
    assert types["homicide"]["last"]["incident_id"] == fatal


def test_homicide_without_shooting_not_a_shooting(client, add):
    shooting = add("shooting", hours_ago=20)
    stabbing = add("homicide", hours_ago=5, was_shooting=False)

    types = by_type(client.get("/timer").json())
    assert types["shooting"]["last"]["incident_id"] == shooting
    assert types["homicide"]["last"]["incident_id"] == stabbing


def test_rejecting_fatal_shooting_updates_shooting_timer(client, add):
    shooting = add("shooting", hours_ago=20)
    fatal = add("homicide", hours_ago=5, was_shooting=True)
    client.post(f"/admin/incidents/{fatal}/reject", headers=AUTH)

    types = by_type(client.get("/timer").json())
    assert types["shooting"]["last"]["incident_id"] == shooting
    assert types["homicide"]["last"] is None


def test_timer_future_incident_clamps_to_zero(client, add):
    add("shooting", hours_ago=-2)
    assert client.get("/timer").json()["overall"]["last"]["seconds_since"] == 0


def test_rejecting_updates_timer(client, add):
    older = add("shooting", hours_ago=5)
    newer = add("shooting", hours_ago=1)
    assert client.get("/timer").json()["overall"]["last"]["incident_id"] == newer

    r = client.post(f"/admin/incidents/{newer}/reject", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"
    assert client.get("/timer").json()["overall"]["last"]["incident_id"] == older

    client.post(f"/admin/incidents/{older}/reject", headers=AUTH)
    body = client.get("/timer").json()
    assert body["overall"]["last"] is None
    assert by_type(body)["shooting"]["last"] is None


def test_confirming_updates_timer_and_incidents(client, add):
    pending = add("homicide", hours_ago=2, status="review")
    assert client.get("/timer").json()["overall"]["last"] is None
    assert client.get("/incidents").json() == []

    r = client.post(f"/admin/incidents/{pending}/confirm", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["status"] == "confirmed"
    assert by_type(client.get("/timer").json())["homicide"]["last"]["incident_id"] == pending
    assert [i["id"] for i in client.get("/incidents").json()] == [pending]


# --- /incidents ---


def test_incidents_lists_confirmed_with_articles(client, add):
    old = add("burglary", hours_ago=30, articles=1)
    new = add("shooting", hours_ago=2, articles=2)
    add("shooting", hours_ago=1, status="review", articles=1)
    add("shooting", hours_ago=1, status="rejected")

    body = client.get("/incidents").json()
    assert [i["id"] for i in body] == [new, old]
    first = body[0]
    assert first["crime_type"] == "shooting"
    assert first["occurred_at"] == "2026-10-05T10:00:00Z"
    assert first["status"] == "confirmed"
    assert first["was_shooting"] is True
    assert first["articles"] == [
        {
            "title": "Story 0",
            "url": f"https://ksdk.example/story/{new}/0",
            "source_name": "KSDK",
            "published_at": "2026-10-05T10:30:00Z",
        },
        {
            "title": "Story 1",
            "url": f"https://ksdk.example/story/{new}/1",
            "source_name": "KSDK",
            "published_at": "2026-10-05T11:30:00Z",
        },
    ]


def test_incidents_limit(client, add):
    for h in range(5):
        add("shooting", hours_ago=h + 1)
    assert len(client.get("/incidents?limit=3").json()) == 3
    assert len(client.get("/incidents").json()) == 5
    assert client.get("/incidents?limit=0").status_code == 422
    assert client.get("/incidents?limit=101").status_code == 422


# --- /stats ---


def test_stats_no_incidents(client):
    body = client.get("/stats").json()
    assert body["overall"]["longest"] is None
    assert all(e["longest"] is None for e in body["by_type"])


def test_stats_gap_calculation(client, add):
    # Shootings at -100h, -60h, -50h; burglary at -90h; homicide at -10h (all confirmed).
    s1 = add("shooting", hours_ago=100)
    b1 = add("burglary", hours_ago=90)
    s2 = add("shooting", hours_ago=60)
    s3 = add("shooting", hours_ago=50)
    h1 = add("homicide", hours_ago=10)
    add("shooting", hours_ago=80, status="rejected")  # would split the 100h->60h gap

    body = client.get("/stats").json()

    # Overall: 100->90 (10h), 90->60 (30h), 60->50 (10h), 50->10 (40h), 10->now (10h).
    assert body["overall"]["longest"] == {
        "seconds": 40 * 3600,
        "start_incident_id": s3,
        "end_incident_id": h1,
        "start": "2026-10-03T10:00:00Z",
        "end": "2026-10-05T02:00:00Z",
        "ongoing": False,
    }
    types = by_type(body)
    # Shootings: 100->60 (40h), 60->50 (10h), 50->now (50h ongoing wins).
    shooting = types["shooting"]["longest"]
    assert shooting["seconds"] == 50 * 3600
    assert shooting["ongoing"] is True
    assert shooting["start_incident_id"] == s3
    assert shooting["end_incident_id"] is None
    assert shooting["end"] == "2026-10-05T12:00:00Z"
    # A single incident: the only gap is the ongoing one.
    assert types["burglary"]["longest"]["seconds"] == 90 * 3600
    assert types["burglary"]["longest"]["start_incident_id"] == b1
    assert types["homicide"]["longest"]["seconds"] == 10 * 3600
    assert s1 and s2  # used only to shape the gaps


def test_stats_fatal_shooting_counts_as_shooting(client, add):
    add("shooting", hours_ago=100)
    fatal = add("homicide", hours_ago=60, was_shooting=True)
    add("shooting", hours_ago=50)
    add("homicide", hours_ago=10, was_shooting=False)  # not a shooting

    types = by_type(client.get("/stats").json())
    # Shootings: 100->60 (40h, ends at the fatal shooting), 60->50, 50->now (50h ongoing).
    assert types["shooting"]["longest"]["seconds"] == 50 * 3600
    assert types["shooting"]["longest"]["ongoing"] is True
    # Homicides: 60->10 (50h), 10->now (10h).
    homicide = types["homicide"]["longest"]
    assert (homicide["seconds"], homicide["start_incident_id"]) == (50 * 3600, fatal)


def test_stats_closed_gap_beats_short_ongoing(client, add):
    a = add("shooting", hours_ago=100)
    b = add("shooting", hours_ago=1)
    longest = client.get("/stats").json()["overall"]["longest"]
    assert longest["seconds"] == 99 * 3600
    assert (longest["start_incident_id"], longest["end_incident_id"]) == (a, b)
    assert longest["ongoing"] is False


# --- admin ---

ADMIN_CALLS = [
    ("get", "/admin/review"),
    ("post", "/admin/incidents/1/confirm"),
    ("post", "/admin/incidents/1/reject"),
    ("post", "/admin/incidents/1/merge?into=2"),
    ("patch", "/admin/incidents/1"),
]


@pytest.mark.parametrize("method,path", ADMIN_CALLS)
@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}, {"X-Admin-Token": ""}])
def test_admin_rejects_missing_or_wrong_token(client, add, method, path, headers):
    add("shooting", status="review")
    r = getattr(client, method)(path, headers=headers)
    assert r.status_code == 401
    # Nothing changed.
    assert client.get("/admin/review", headers=AUTH).json()[0]["status"] == "review"


@pytest.mark.parametrize("method,path", ADMIN_CALLS)
@pytest.mark.parametrize("configured", [None, ""])
def test_admin_disabled_without_configured_token(client, method, path, configured):
    app.dependency_overrides[get_admin_token] = lambda: configured
    r = getattr(client, method)(path, headers={"X-Admin-Token": ""})
    assert r.status_code == 503


def test_admin_review_lists_review_incidents(client, add):
    later = add("shooting", hours_ago=1, status="review", articles=1)
    earlier = add("burglary", hours_ago=5, status="review")
    add("homicide", hours_ago=2, status="confirmed")
    add("homicide", hours_ago=3, status="rejected")

    body = client.get("/admin/review", headers=AUTH).json()
    assert [i["id"] for i in body] == [earlier, later]
    assert body[1]["articles"][0]["source_name"] == "KSDK"


def test_admin_unknown_incident_404(client):
    assert client.post("/admin/incidents/999/confirm", headers=AUTH).status_code == 404
    assert client.post("/admin/incidents/999/reject", headers=AUTH).status_code == 404


def test_admin_token_read_from_settings(monkeypatch):
    from config import get_settings

    monkeypatch.setenv("ADMIN_TOKEN", "from-env")
    get_settings.cache_clear()
    try:
        assert get_admin_token() == "from-env"
    finally:
        get_settings.cache_clear()


# --- page ---


def test_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    html = r.text
    assert "without a shooting, burglary, or killing" in html
    assert "Based on local news reports and St. Louis police (SLMPD) news releases." in html
    assert "Sources: KSDK 5, Fox2, KMOV 4, St. Louis Post-Dispatch, St. Louis Public Radio, SLMPD." in html
    assert "America/Chicago" in html
    for path in ("/timer", "/stats", "/incidents", "/health"):
        assert path in html


# --- admin merge ---


def merge(client, source, target):
    return client.post(f"/admin/incidents/{source}/merge?into={target}", headers=AUTH)


def article_urls(incident_json) -> set[str]:
    return {a["url"] for a in incident_json["articles"]}


def test_merge_moves_links_and_combines_fields(client, add, session_factory):
    target = add("homicide", hours_ago=10, status="review", location=None, articles=1,
                 was_shooting=False)
    source = add("shooting", hours_ago=12, status="confirmed", location="West End",
                 neighborhood="West End", articles=2)

    r = merge(client, source, target)

    assert r.status_code == 200
    body = r.json()
    assert body["id"] == target
    assert len(body["articles"]) == 3
    assert body["crime_type"] == "homicide"
    assert body["was_shooting"] is True  # ORed in from the shooting
    assert body["status"] == "confirmed"  # either was confirmed
    assert body["occurred_at"] == "2026-10-05T00:00:00Z"  # earliest reported time
    assert (body["location"], body["neighborhood"]) == ("West End", "West End")
    with session_factory() as s:
        src = s.get(Incident, source)
        assert (src.status, src.merged_into_id) == ("merged", target)
        assert src.raw_items == []


def test_merge_reported_time_beats_earlier_estimate(client, add):
    target = add("shooting", hours_ago=10)
    source = add("shooting", hours_ago=20, time_estimated=True)
    body = merge(client, source, target).json()
    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-05T02:00:00Z", False)


def test_merge_takes_reported_time_from_source(client, add):
    target = add("shooting", hours_ago=20, time_estimated=True)
    source = add("shooting", hours_ago=10)
    body = merge(client, source, target).json()
    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-05T02:00:00Z", False)


def test_merge_both_estimated_keeps_earliest(client, add):
    target = add("shooting", hours_ago=10, time_estimated=True)
    source = add("shooting", hours_ago=20, time_estimated=True)
    body = merge(client, source, target).json()
    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-04T16:00:00Z", True)


def test_merge_never_downgrades_confirmed_target(client, add):
    target = add("shooting", hours_ago=10, status="confirmed")
    source = add("shooting", hours_ago=11, status="review")
    assert merge(client, source, target).json()["status"] == "confirmed"


def test_merge_updates_timer_and_lists(client, add):
    target = add("homicide", hours_ago=30, was_shooting=True)
    source = add("homicide", hours_ago=2, was_shooting=True)  # duplicate, later time
    assert client.get("/timer").json()["overall"]["last"]["incident_id"] == source

    merge(client, source, target)

    timer = client.get("/timer").json()
    assert timer["overall"]["last"]["incident_id"] == target
    assert timer["overall"]["last"]["seconds_since"] == 30 * 3600
    assert [i["id"] for i in client.get("/incidents").json()] == [target]
    stats = client.get("/stats").json()
    assert stats["overall"]["longest"]["start_incident_id"] == target
    assert stats["overall"]["longest"]["ongoing"] is True


def test_merge_rejected_source_into_live_target(client, add):
    target = add("shooting", hours_ago=5, articles=1)
    source = add("shooting", hours_ago=6, status="rejected", articles=1)
    body = merge(client, source, target).json()
    assert len(body["articles"]) == 2


def test_merge_skips_item_already_linked_to_target(client, add, session_factory):
    target = add("shooting", hours_ago=5, articles=1)
    source = add("shooting", hours_ago=6, articles=1)
    with session_factory() as s:
        shared = s.scalars(
            select(IncidentItem.raw_item_id).where(IncidentItem.incident_id == target)
        ).one()
        s.add(IncidentItem(incident_id=source, raw_item_id=shared))
        s.commit()
    body = merge(client, source, target).json()
    assert len(body["articles"]) == 2


@pytest.mark.parametrize(
    "setup, expected",
    [
        ("self", 400),
        ("missing_source", 404),
        ("missing_target", 404),
        ("source_merged", 409),
        ("target_merged", 409),
        ("target_rejected", 409),
        ("incompatible", 409),
    ],
)
def test_merge_errors(client, add, session_factory, setup, expected):
    a = add("shooting", hours_ago=5)
    b = add("burglary" if setup == "incompatible" else "shooting", hours_ago=6,
            status="rejected" if setup == "target_rejected" else "confirmed")
    source, target = {
        "self": (a, a),
        "missing_source": (999, b),
        "missing_target": (a, 999),
    }.get(setup, (a, b))
    if setup in ("source_merged", "target_merged"):
        with session_factory() as s:
            s.get(Incident, a if setup == "source_merged" else b).status = "merged"
            s.commit()

    r = merge(client, source, target)

    assert r.status_code == expected
    with session_factory() as s:  # nothing changed
        assert s.get(Incident, a).merged_into_id is None


def test_merge_requires_into(client, add):
    a = add("shooting")
    r = client.post(f"/admin/incidents/{a}/merge", headers=AUTH)
    assert r.status_code == 422


# --- admin edit ---


def edit(client, incident_id, body):
    return client.patch(f"/admin/incidents/{incident_id}", json=body, headers=AUTH)


def test_edit_time_marks_it_manual(client, add, session_factory):
    inc = add("burglary", hours_ago=12, time_estimated=True, articles=1)

    # Naive = St. Louis time: Sunday Oct 4, 04:00 CDT.
    r = edit(client, inc, {"occurred_at": "2026-10-04T04:00:00", "time_estimated": True})

    assert r.status_code == 200
    body = r.json()
    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-04T09:00:00Z", True)
    assert len(body["articles"]) == 1
    with session_factory() as s:
        i = s.get(Incident, inc)
        assert (i.manual_occurred_at, i.manual_location, i.manual_neighborhood) == (
            True, False, False,
        )


def test_edit_time_with_offset_and_only_time_estimated(client, add, session_factory):
    inc = add("shooting", hours_ago=3, time_estimated=True)
    body = edit(client, inc, {"occurred_at": "2026-10-05T01:30:00-05:00"}).json()
    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-05T06:30:00Z", True)

    other = add("shooting", hours_ago=5, time_estimated=True)
    body = edit(client, other, {"time_estimated": False}).json()
    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-05T07:00:00Z", False)
    with session_factory() as s:
        assert s.get(Incident, other).manual_occurred_at is True


def test_edit_location_and_neighborhood(client, add, session_factory):
    inc = add("burglary", location="somewhere", neighborhood=None)

    body = edit(client, inc, {"location": "  9th St. and Allen Ave  ",
                              "neighborhood": "skinker-debaliviere"}).json()
    assert (body["location"], body["neighborhood"]) == (
        "9th St. and Allen Ave", "Skinker DeBaliviere",
    )
    body = edit(client, inc, {"location": None, "neighborhood": None}).json()
    assert (body["location"], body["neighborhood"]) == (None, None)
    with session_factory() as s:
        i = s.get(Incident, inc)
        assert (i.manual_occurred_at, i.manual_location, i.manual_neighborhood) == (
            False, True, True,
        )


def test_edit_moves_timer(client, add):
    older = add("burglary", hours_ago=30)
    newer = add("burglary", hours_ago=2)
    assert client.get("/timer").json()["overall"]["last"]["incident_id"] == newer

    edit(client, newer, {"occurred_at": "2026-10-03T12:00:00Z"})

    assert client.get("/timer").json()["overall"]["last"]["incident_id"] == older


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"occurred_at": None},
        {"occurred_at": "not a date"},
        {"time_estimated": None},
        {"neighborhood": "Atlantis"},
        {"status": "confirmed"},
    ],
)
def test_edit_rejects_bad_bodies(client, add, session_factory, body):
    inc = add("shooting", hours_ago=1, neighborhood="Shaw")
    r = edit(client, inc, body)
    assert r.status_code == 422
    with session_factory() as s:
        i = s.get(Incident, inc)
        assert (i.neighborhood, i.status, i.manual_occurred_at) == ("Shaw", "confirmed", False)


def test_edit_missing_or_merged_incident(client, add, session_factory):
    assert edit(client, 999, {"location": "x"}).status_code == 404
    a = add("shooting", hours_ago=1)
    b = add("shooting", hours_ago=2)
    merge(client, a, b)
    r = edit(client, a, {"location": "x"})
    assert r.status_code == 409
    assert f"merged into {b}" in r.json()["detail"]


def test_merge_keeps_manual_fields(client, add, session_factory):
    target = add("shooting", hours_ago=10, location=None)
    source = add("shooting", hours_ago=20, location="Grand Blvd")
    edit(client, target, {"occurred_at": "2026-10-05T03:00:00Z", "location": None})

    body = merge(client, source, target).json()

    # The source's earlier reported time and its location don't override the edits.
    assert (body["occurred_at"], body["location"]) == ("2026-10-05T03:00:00Z", None)


def test_merge_carries_manual_time_from_source(client, add, session_factory):
    target = add("shooting", hours_ago=10)
    source = add("shooting", hours_ago=20, time_estimated=True)
    edit(client, source, {"occurred_at": "2026-10-05T05:00:00Z"})

    body = merge(client, source, target).json()

    assert (body["occurred_at"], body["time_estimated"]) == ("2026-10-05T05:00:00Z", True)
    with session_factory() as s:
        assert s.get(Incident, target).manual_occurred_at is True
