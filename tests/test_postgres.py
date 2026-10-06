"""Migrations and the SQLite -> Postgres copy on a real Postgres database. Skipped unless
DATABASE_URL_TEST is set (see tests/pg.py). The rest of the suite also runs on
Postgres then, through the `session` / `session_factory` fixtures."""

from datetime import datetime, timedelta, timezone

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import Base, make_engine
from models import Incident, PipelineRun, RawItem, Source
from scripts.copy_sqlite_to_postgres import copy_database, head_revision
from tests import pg
from tests.dbhelpers import add_sample_data, run_alembic, snapshot

pytestmark = pg.requires_pg

APP_TABLES = {
    "sources", "raw_items", "classifications", "incidents", "incident_items",
    "pipeline_runs", "job_locks", "alerts_sent",
}


@pytest.fixture
def pg_url():
    """URL of a fresh, empty schema (dropped afterwards)."""
    with pg.temp_schema() as url:
        yield url


@pytest.fixture
def migrated(pg_url, monkeypatch):
    run_alembic(monkeypatch, pg_url)
    engine = make_engine(pg_url)
    yield engine
    engine.dispose()


def test_upgrade_head_creates_schema(migrated):
    insp = inspect(migrated)
    assert set(insp.get_table_names()) == APP_TABLES | {"alembic_version"}
    with migrated.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() \
            == head_revision()
    columns = {c["name"]: c for c in insp.get_columns("incidents")}
    assert {"time_estimated", "was_shooting", "neighborhood", "merged_into_id"} <= set(columns)
    # UTCDateTime is timestamptz on Postgres.
    assert columns["occurred_at"]["type"].timezone is True


def test_migrations_match_models(migrated):
    """Autogenerate finds nothing to do: the migrated schema is what models.py says."""
    with migrated.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_check_constraints_enforced(migrated):
    with Session(migrated) as s:
        s.add(Incident(crime_type="arson", occurred_at=datetime.now(timezone.utc)))
        with pytest.raises(IntegrityError):
            s.commit()


def test_merged_status_allowed(migrated):
    with Session(migrated) as s:
        target = Incident(crime_type="homicide", occurred_at=datetime.now(timezone.utc))
        s.add(target)
        s.flush()
        s.add(Incident(crime_type="shooting", occurred_at=datetime.now(timezone.utc),
                       status="merged", merged_into_id=target.id))
        s.commit()


def test_downgrade_to_base_and_back(migrated, pg_url, monkeypatch):
    with Session(migrated) as s:
        add_sample_data(s)
    migrated.dispose()

    run_alembic(monkeypatch, pg_url, "downgrade", "base")
    engine = make_engine(pg_url)
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    engine.dispose()

    run_alembic(monkeypatch, pg_url)
    engine = make_engine(pg_url)
    assert APP_TABLES <= set(inspect(engine).get_table_names())
    engine.dispose()


def test_data_migrations_backfill(pg_url, monkeypatch):
    """Rows written before was_shooting / time_precision get backfilled on Postgres."""
    run_alembic(monkeypatch, pg_url, revision="e4d4a3446628")
    engine = make_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO sources (id, name, url, type) VALUES (1, 'S', 'https://s', 'rss')"
        ))
        for n in (1, 2, 3):
            conn.execute(text(
                "INSERT INTO raw_items (id, source_id, url, url_hash, title, status) "
                f"VALUES ({n}, 1, 'https://s/{n}', 'h{n}', 't', 'classified')"
            ))
        rows = [
            # id, raw_item, crime_type, occurred_at
            (1, 1, "shooting", "2026-10-04 05:40:00+00"),  # 00:40 St. Louis -> exact
            (2, 2, "homicide", "2026-10-04 05:00:00+00"),  # midnight St. Louis -> date_only
            (3, 3, "burglary", None),                      # -> unknown
        ]
        for cid, rid, crime, occurred in rows:
            conn.execute(
                text(
                    "INSERT INTO classifications (id, raw_item_id, is_crime, crime_type, "
                    "in_stl, occurred_at, confidence, model, prompt_version) VALUES "
                    "(:id, :rid, true, :crime, true, :occ, 0.9, 'm', 'v1')"
                ),
                {"id": cid, "rid": rid, "crime": crime, "occ": occurred},
            )
        conn.execute(text(
            "INSERT INTO incidents (id, crime_type, occurred_at, status) "
            "VALUES (1, 'homicide', '2026-10-04 05:40:00+00', 'confirmed')"
        ))
        conn.execute(text("INSERT INTO incident_items VALUES (1, 1)"))
    engine.dispose()

    run_alembic(monkeypatch, pg_url)
    engine = make_engine(pg_url)
    with engine.connect() as conn:
        shooting = dict(conn.execute(text(
            "SELECT id, was_shooting FROM classifications ORDER BY id"
        )).all())
        precision = dict(conn.execute(text(
            "SELECT id, time_precision FROM classifications ORDER BY id"
        )).all())
        incident_shot = conn.execute(text(
            "SELECT was_shooting FROM incidents WHERE id = 1"
        )).scalar()
    engine.dispose()
    assert shooting == {1: True, 2: False, 3: False}
    assert precision == {1: "exact", 2: "date_only", 3: "unknown"}
    assert incident_shot is True


def test_copy_sqlite_to_postgres(tmp_path, monkeypatch, migrated):
    sqlite_url = f"sqlite:///{(tmp_path / 'local.db').as_posix()}"
    run_alembic(monkeypatch, sqlite_url)
    source = create_engine(sqlite_url)
    with Session(source) as s:
        add_sample_data(s)

    counts = copy_database(source, migrated)

    assert counts["raw_items"] == 3 and counts["incidents"] == 2
    assert snapshot(migrated) == snapshot(source)
    # Sequences moved past the copied ids: new rows don't collide.
    with Session(migrated) as s:
        s.add(Source(name="New", url="https://example.com/new", type="rss"))
        s.add(Incident(crime_type="burglary", occurred_at=datetime.now(timezone.utc)))
        s.add(PipelineRun(started_at=datetime.now(timezone.utc)))
        s.commit()
        assert s.scalar(select(Source.id).where(Source.name == "New")) == 2
        assert s.scalar(select(Incident.id).where(Incident.crime_type == "burglary")) == 3
        new_item = RawItem(source_id=1, url="https://x", url_hash="x", title="t")
        s.add(new_item)
        s.commit()
        assert new_item.id == 4
    source.dispose()


def test_datetimes_round_trip_as_utc(migrated):
    chicago_naive = datetime(2026, 7, 4, 21, 30)  # naive = St. Louis time (CDT, UTC-5)
    aware = datetime(2026, 1, 15, 3, 0, tzinfo=timezone(timedelta(hours=2)))
    with Session(migrated) as s:
        a = Incident(crime_type="shooting", occurred_at=chicago_naive)
        b = Incident(crime_type="shooting", occurred_at=aware)
        s.add_all([a, b])
        s.commit()
        ids = (a.id, b.id)
    with Session(migrated) as s:
        got = [s.get(Incident, i).occurred_at for i in ids]
    assert got == [
        datetime(2026, 7, 5, 2, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 15, 1, 0, tzinfo=timezone.utc),
    ]
    assert all(dt.tzinfo == timezone.utc for dt in got)
