import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import models  # noqa: F401  (registers tables on Base.metadata)
from db import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture
def session_factory():
    """Factory for many sessions over one in-memory database."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


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
