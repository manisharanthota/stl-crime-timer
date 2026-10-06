from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from config import get_settings
from models import Classification, RawItem, Source
from timeutil import LOCAL_TZ, to_local, to_utc

ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc


# --- to_utc / to_local -------------------------------------------------------

@pytest.mark.parametrize(
    "naive_local, expected_utc",
    [
        # Winter, CST (UTC-6)
        (datetime(2026, 1, 15, 21, 0), datetime(2026, 1, 16, 3, 0, tzinfo=UTC)),
        # Summer, CDT (UTC-5)
        (datetime(2026, 7, 4, 21, 0), datetime(2026, 7, 5, 2, 0, tzinfo=UTC)),
        # Spring-forward day, 2026-03-08: before and after the 2:00 jump
        (datetime(2026, 3, 8, 1, 30), datetime(2026, 3, 8, 7, 30, tzinfo=UTC)),
        (datetime(2026, 3, 8, 3, 30), datetime(2026, 3, 8, 8, 30, tzinfo=UTC)),
        # Fall-back day, 2026-11-01: 1:30 happens twice; fold=0 is the first (CDT)
        (datetime(2026, 11, 1, 1, 30), datetime(2026, 11, 1, 6, 30, tzinfo=UTC)),
        (datetime(2026, 11, 1, 1, 30, fold=1), datetime(2026, 11, 1, 7, 30, tzinfo=UTC)),
    ],
)
def test_to_utc_treats_naive_as_st_louis(naive_local, expected_utc):
    result = to_utc(naive_local)
    assert result == expected_utc
    assert result.tzinfo is UTC


def test_to_utc_converts_aware_values():
    value = datetime(2026, 7, 4, 21, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert to_utc(value) == datetime(2026, 7, 5, 2, 0, tzinfo=UTC)
    assert to_utc(value).tzinfo is UTC


@pytest.mark.parametrize(
    "utc, local_hour, offset_hours",
    [
        (datetime(2026, 1, 16, 3, 0, tzinfo=UTC), 21, -6),  # CST
        (datetime(2026, 7, 5, 2, 0, tzinfo=UTC), 21, -5),   # CDT
        (datetime(2026, 3, 8, 7, 59, tzinfo=UTC), 1, -6),   # just before spring forward
        (datetime(2026, 3, 8, 8, 0, tzinfo=UTC), 3, -5),    # just after
    ],
)
def test_to_local_for_display(utc, local_hour, offset_hours):
    local = to_local(utc)
    assert local.tzinfo == LOCAL_TZ
    assert local.hour == local_hour
    assert local.utcoffset() == timedelta(hours=offset_hours)


def test_to_local_treats_naive_as_utc():
    assert to_local(datetime(2026, 7, 5, 2, 0)).hour == 21


# --- UTCDateTime column ------------------------------------------------------

@pytest.fixture
def item(session):
    source = Source(name="S", url="https://s.example/feed", type="news")
    session.add(source)
    session.flush()
    item = RawItem(source_id=source.id, url="https://s.example/1", url_hash="h1", title="T")
    session.add(item)
    session.commit()
    return item


def add_classification(session, item, occurred_at):
    c = Classification(
        raw_item_id=item.id, is_crime=True, crime_type="shooting", in_stl=True,
        occurred_at=occurred_at, confidence=0.9, model="m", prompt_version="v2",
    )
    session.add(c)
    session.commit()
    return c


def stored(session, sql):
    return session.execute(text(sql)).scalar()


def assert_stored_utc(session, sql, expected: datetime) -> None:
    """The raw stored value is that UTC instant: naive UTC text on SQLite, a
    timestamptz (aware datetime) on Postgres."""
    raw = stored(session, sql)
    if isinstance(raw, str):
        assert raw.startswith(expected.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        assert raw.tzinfo is not None and raw == expected


@pytest.mark.parametrize(
    "value, expected_utc",
    [
        # Naive input = St. Louis time; summer (CDT) and winter (CST)
        (datetime(2026, 7, 4, 21, 0), datetime(2026, 7, 5, 2, 0, tzinfo=UTC)),
        (datetime(2026, 1, 15, 21, 0), datetime(2026, 1, 16, 3, 0, tzinfo=UTC)),
        # Spring-forward day
        (datetime(2026, 3, 8, 3, 30), datetime(2026, 3, 8, 8, 30, tzinfo=UTC)),
        # Aware input in another offset
        (datetime(2026, 7, 4, 21, 0, tzinfo=timezone(timedelta(hours=-5))),
         datetime(2026, 7, 5, 2, 0, tzinfo=UTC)),
    ],
)
def test_datetimes_stored_as_utc_and_read_back_aware(session, item, value, expected_utc):
    c = add_classification(session, item, value)
    session.expire_all()

    assert c.occurred_at == expected_utc
    assert c.occurred_at.tzinfo is UTC
    assert_stored_utc(session, "SELECT occurred_at FROM classifications", expected_utc)


def test_all_datetime_columns_are_utc(session, item):
    item.published_at = datetime(2026, 7, 4, 12, 0)  # naive St. Louis -> 17:00 UTC
    item.source.last_success_at = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
    session.commit()
    session.expire_all()

    assert item.published_at == datetime(2026, 7, 4, 17, 0, tzinfo=UTC)
    assert item.source.last_success_at == datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
    # Server-default created_at (SQLite CURRENT_TIMESTAMP is UTC; Postgres now() is
    # timestamptz) reads back aware UTC.
    assert item.created_at.tzinfo is UTC
    assert abs(datetime.now(UTC) - item.created_at) < timedelta(minutes=1)
    assert_stored_utc(
        session, "SELECT published_at FROM raw_items", datetime(2026, 7, 4, 17, 0, tzinfo=UTC)
    )


def test_filtering_compares_in_utc(session, item):
    add_classification(session, item, datetime(2026, 7, 5, 2, 0, tzinfo=UTC))
    # 20:59 St. Louis (CDT) on Jul 4 is 01:59 UTC Jul 5, just before the stored time.
    cutoff = datetime(2026, 7, 4, 20, 59)
    assert session.query(Classification).filter(Classification.occurred_at > cutoff).count() == 1
    cutoff = datetime(2026, 7, 4, 21, 1)
    assert session.query(Classification).filter(Classification.occurred_at > cutoff).count() == 0


# --- migration of existing data ----------------------------------------------

def test_migration_converts_local_occurred_at_to_utc(tmp_path, monkeypatch):
    db_url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    get_settings.cache_clear()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    engine = create_engine(db_url)
    try:
        command.upgrade(cfg, "c62cefc9aa3b")
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO sources (id, name, url, type) VALUES (1, 'S', 'u', 'news')"))
            conn.execute(text(
                "INSERT INTO raw_items (id, source_id, url, url_hash, title) "
                "VALUES (1, 1, 'u1', 'h1', 'T'), (2, 1, 'u2', 'h2', 'T')"
            ))
            # Old rows: St. Louis wall-clock times with the offset dropped.
            conn.execute(text(
                "INSERT INTO classifications (raw_item_id, is_crime, in_stl, occurred_at, "
                "confidence, model, prompt_version) VALUES "
                "(1, 1, 1, '2026-10-02 00:40:00.000000', 0.99, 'm', 'v2'), "  # CDT
                "(2, 1, 1, '2026-01-15 21:00:00.000000', 0.99, 'm', 'v2')"    # CST
            ))

        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT occurred_at FROM classifications ORDER BY id")).scalars().all()
        assert rows == ["2026-10-02 05:40:00.000000", "2026-01-16 03:00:00.000000"]

        command.downgrade(cfg, "c62cefc9aa3b")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT occurred_at FROM classifications ORDER BY id")).scalars().all()
        assert rows == ["2026-10-02 00:40:00.000000", "2026-01-15 21:00:00.000000"]
    finally:
        engine.dispose()
        get_settings.cache_clear()
