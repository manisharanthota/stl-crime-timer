from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from config import get_settings

ROOT = Path(__file__).resolve().parent.parent


def test_upgrade_head_creates_schema(tmp_path, monkeypatch):
    db_url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    get_settings.cache_clear()
    try:
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "alembic"))
        command.upgrade(cfg, "head")
    finally:
        get_settings.cache_clear()

    engine = create_engine(db_url)
    insp = inspect(engine)
    assert set(insp.get_table_names()) >= {
        "sources",
        "raw_items",
        "classifications",
        "incidents",
        "incident_items",
        "pipeline_runs",
        "job_locks",
    }
    assert "ix_raw_items_status" in {i["name"] for i in insp.get_indexes("raw_items")}
    assert "ix_incidents_occurred_at" in {
        i["name"] for i in insp.get_indexes("incidents")
    }
    assert "ix_pipeline_runs_started_at" in {
        i["name"] for i in insp.get_indexes("pipeline_runs")
    }
    assert {"time_estimated", "was_shooting"} <= {c["name"] for c in insp.get_columns("incidents")}
    assert "was_shooting" in {c["name"] for c in insp.get_columns("classifications")}
    engine.dispose()


def _alembic(tmp_path, monkeypatch) -> tuple[Config, str]:
    db_url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    get_settings.cache_clear()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return cfg, db_url


def test_was_shooting_backfill(tmp_path, monkeypatch):
    cfg, db_url = _alembic(tmp_path, monkeypatch)
    try:
        command.upgrade(cfg, "e4d4a3446628")  # just before was_shooting
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO sources (id, name, url, type, active, fail_count)"
                " VALUES (1, 's', 'https://s.example', 'rss', 1, 0)"
            ))
            for i in range(1, 5):
                conn.execute(text(
                    "INSERT INTO raw_items (id, source_id, url, url_hash, title, status, retries)"
                    f" VALUES ({i}, 1, 'https://s.example/{i}', 'h{i}', 't', 'classified', 0)"
                ))
            for cid, item, crime_type in [
                (1, 1, "shooting"),
                (2, 2, "homicide"),
                (3, 3, "burglary"),
                (4, 4, "shooting"),
                (5, 4, "homicide"),  # newest classification of item 4 wins
            ]:
                conn.execute(text(
                    "INSERT INTO classifications (id, raw_item_id, is_crime, crime_type, in_stl,"
                    " confidence, model, prompt_version)"
                    f" VALUES ({cid}, {item}, 1, '{crime_type}', 1, 0.9, 'm', 'v2')"
                ))
            for iid, crime_type, items in [
                (1, "shooting", []),       # a shooting, no links needed
                (2, "homicide", [1, 2]),   # shooting -> homicide upgrade
                (3, "homicide", [2]),      # homicide, no sign of a gun
                (4, "burglary", [3]),
                (5, "homicide", [4]),      # reclassified away from shooting
            ]:
                conn.execute(text(
                    "INSERT INTO incidents (id, crime_type, occurred_at, status, time_estimated)"
                    f" VALUES ({iid}, '{crime_type}', '2026-10-01 12:00:00', 'confirmed', 0)"
                ))
                for item in items:
                    conn.execute(text(
                        f"INSERT INTO incident_items VALUES ({iid}, {item})"
                    ))
        engine.dispose()

        command.upgrade(cfg, "head")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            classified = dict(conn.execute(
                text("SELECT id, was_shooting FROM classifications")
            ).all())
            incidents = dict(conn.execute(
                text("SELECT id, was_shooting FROM incidents")
            ).all())
        engine.dispose()
        assert classified == {1: 1, 2: 0, 3: 0, 4: 1, 5: 0}
        assert incidents == {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}

        command.downgrade(cfg, "e4d4a3446628")
        engine = create_engine(db_url)
        assert "was_shooting" not in {c["name"] for c in inspect(engine).get_columns("incidents")}
        engine.dispose()
    finally:
        get_settings.cache_clear()


def test_downgrade_to_pipeline_runs_with_linked_incidents(tmp_path, monkeypatch):
    """Dropping incidents columns must not rebuild the table: with SQLite FKs on,
    incident_items rows referencing incidents would make the rebuild fail."""
    cfg, db_url = _alembic(tmp_path, monkeypatch)
    try:
        command.upgrade(cfg, "head")
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO sources (id, name, url, type) VALUES (1, 's', 'https://s.example', 'rss')"
            ))
            conn.execute(text(
                "INSERT INTO raw_items (id, source_id, url, url_hash, title)"
                " VALUES (1, 1, 'https://s.example/1', 'h1', 't')"
            ))
            conn.execute(text(
                "INSERT INTO incidents (id, crime_type, occurred_at)"
                " VALUES (1, 'shooting', '2026-10-01 12:00:00')"
            ))
            conn.execute(text("INSERT INTO incident_items VALUES (1, 1)"))
        engine.dispose()

        command.downgrade(cfg, "5d1f7a3b9c20")  # before time_estimated

        engine = create_engine(db_url)
        columns = {c["name"] for c in inspect(engine).get_columns("incidents")}
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM incident_items")).scalar() == 1
        engine.dispose()
        assert not {"time_estimated", "was_shooting"} & columns
    finally:
        get_settings.cache_clear()
