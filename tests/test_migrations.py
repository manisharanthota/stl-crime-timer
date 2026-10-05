from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

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
    }
    assert "ix_raw_items_status" in {i["name"] for i in insp.get_indexes("raw_items")}
    assert "ix_incidents_occurred_at" in {
        i["name"] for i in insp.get_indexes("incidents")
    }
    assert "time_estimated" in {c["name"] for c in insp.get_columns("incidents")}
    engine.dispose()
