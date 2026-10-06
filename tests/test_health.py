from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_now, get_session
from models import PipelineRun

NOW = datetime(2026, 10, 4, 13, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(session_factory):
    def override():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = override
    app.dependency_overrides[get_now] = lambda: NOW
    yield TestClient(app)
    app.dependency_overrides.clear()


def add_run(session_factory, started_at, finished_at=None, status="success"):
    with session_factory() as s:
        s.add(PipelineRun(started_at=started_at, finished_at=finished_at, status=status))
        s.commit()


def ago(minutes: float) -> datetime:
    return NOW - timedelta(minutes=minutes)


def test_health_no_runs_is_stale(client):
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "stale", "last_run": None, "last_success_at": None}


def test_health_recent_success_is_ok(client, session_factory):
    add_run(session_factory, ago(12), ago(10))

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "last_run": {
            "started_at": "2026-10-04T12:48:00Z",
            "finished_at": "2026-10-04T12:50:00Z",
            "status": "success",
        },
        "last_success_at": "2026-10-04T12:50:00Z",
    }


def test_health_old_success_is_stale(client, session_factory):
    add_run(session_factory, ago(32), ago(31))

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "stale"
    assert body["last_success_at"] == "2026-10-04T12:29:00Z"


def test_health_window_boundary(client, session_factory):
    add_run(session_factory, ago(31), ago(30))
    assert client.get("/health").status_code == 503


def test_health_recent_partial_after_old_success_is_stale(client, session_factory):
    add_run(session_factory, ago(60), ago(59))
    add_run(session_factory, ago(5), ago(4), status="partial")

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["last_run"]["status"] == "partial"
    assert body["last_success_at"] == "2026-10-04T12:01:00Z"


def test_health_shows_last_run_even_when_not_success(client, session_factory):
    add_run(session_factory, ago(20), ago(19))
    add_run(session_factory, ago(5), ago(3.5), status="failed")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["last_run"] == {
        "started_at": "2026-10-04T12:55:00Z",
        "finished_at": "2026-10-04T12:56:30Z",
        "status": "failed",
    }


def test_health_shows_running_run(client, session_factory):
    add_run(session_factory, ago(15), ago(14))
    add_run(session_factory, ago(1), status="running")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["last_run"] == {
        "started_at": "2026-10-04T12:59:00Z",
        "finished_at": None,
        "status": "running",
    }


def test_health_head(client, session_factory):
    """Uptime monitors check with HEAD; it must give the same status as GET."""
    assert client.head("/health").status_code == 503
    add_run(session_factory, ago(2), ago(1))
    response = client.head("/health")
    assert response.status_code == 200
    assert response.content == b""
