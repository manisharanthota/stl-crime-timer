import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import models  # noqa: F401  (registers tables on Base.metadata)
from db import Base, make_engine
from tests import pg

# Every test using `session` / `session_factory` runs on in-memory SQLite, and again
# on Postgres when DATABASE_URL_TEST is set.
BACKENDS = ["sqlite", "postgres"] if pg.PG_URL else ["sqlite"]


def pytest_sessionstart(session):
    pg.check_safe()


@pytest.fixture(scope="session")
def pg_engine():
    """One Postgres schema with the model tables for the whole run; emptied after
    every test that used it."""
    with pg.temp_schema() as url:
        engine = make_engine(url)
        Base.metadata.create_all(engine)
        yield engine
        engine.dispose()


@pytest.fixture(params=BACKENDS)
def backend_engine(request):
    if request.param == "sqlite":
        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(engine)
        yield engine
        engine.dispose()
    else:
        engine = request.getfixturevalue("pg_engine")
        yield engine
        pg.truncate_all(engine, [t.name for t in Base.metadata.sorted_tables])


@pytest.fixture
def session(backend_engine):
    with Session(backend_engine) as s:
        yield s


@pytest.fixture
def session_factory(backend_engine):
    """Factory for many sessions over one database."""
    yield sessionmaker(bind=backend_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture(autouse=True)
def no_real_alerts(monkeypatch):
    """.env may hold a real Discord webhook: tests run with ALERT_WEBHOOK_URL unset
    (log-only), and any webhook POST through a client a test didn't inject fails."""
    from config import get_settings

    monkeypatch.setenv("ALERT_WEBHOOK_URL", "")
    monkeypatch.setenv("ALERT_ON_NEW_INCIDENT", "")
    monkeypatch.setenv("ALERT_COOLDOWN_HOURS", "")

    def _blocked(*args, **kwargs):
        raise AssertionError("tests must not create a real alert webhook client")

    monkeypatch.setattr("alerts.notify.make_client", _blocked)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
