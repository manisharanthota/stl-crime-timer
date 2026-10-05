from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

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
    ):
        with session_factory() as s:
            incident = Incident(
                crime_type=crime_type,
                occurred_at=NOW - timedelta(hours=hours_ago),
                location=location,
                status=status,
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
    }
    types = by_type(body)
    assert types["shooting"]["last"]["incident_id"] == shooting
    assert types["shooting"]["last"]["seconds_since"] == 10 * 3600
    assert types["burglary"]["last"]["incident_id"] == burglary
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
    assert "Based on local news reports. Not official police data." in html
    assert "America/Chicago" in html
    for path in ("/timer", "/stats", "/incidents", "/health"):
        assert path in html
