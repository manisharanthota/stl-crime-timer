from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_session
from models import PipelineRun


@pytest.fixture
def client(session_factory):
    def override():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_no_runs(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "last_run": None}


def test_health_shows_last_run(client, session_factory):
    with session_factory() as s:
        s.add_all([
            PipelineRun(
                started_at=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
                finished_at=datetime(2026, 10, 4, 12, 1, tzinfo=timezone.utc),
                status="failed",
            ),
            PipelineRun(
                started_at=datetime(2026, 10, 4, 12, 10, tzinfo=timezone.utc),
                finished_at=datetime(2026, 10, 4, 12, 11, 30, tzinfo=timezone.utc),
                status="partial",
            ),
        ])
        s.commit()

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "last_run": {
            "started_at": "2026-10-04T12:10:00Z",
            "finished_at": "2026-10-04T12:11:30Z",
            "status": "partial",
        },
    }


def test_health_shows_running_run(client, session_factory):
    with session_factory() as s:
        s.add(PipelineRun(started_at=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)))
        s.commit()

    last = client.get("/health").json()["last_run"]
    assert last == {"started_at": "2026-10-04T12:00:00Z", "finished_at": None, "status": "running"}
