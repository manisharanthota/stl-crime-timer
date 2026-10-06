"""Helpers shared by the migration, copy-script, and Postgres tests."""

from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config

from config import get_settings
from models import (
    AlertSent,
    Classification,
    Incident,
    IncidentItem,
    JobLock,
    PipelineRun,
    RawItem,
    Source,
    hash_url,
)

ROOT = Path(__file__).resolve().parent.parent


def alembic_config() -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return cfg


def run_alembic(monkeypatch, url: str, action: str = "upgrade", revision: str = "head"):
    """Run `alembic upgrade|downgrade <revision>` against url (env.py reads the URL
    from DATABASE_URL via get_settings)."""
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        getattr(command, action)(alembic_config(), revision)
    finally:
        get_settings.cache_clear()


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def add_sample_data(session) -> None:
    """One of everything, including an incident merged into a later id (self-FK) and
    JSON counts, so copies and round trips have something to get wrong."""
    source = Source(name="KSDK", url="https://example.com/rss", type="rss", fail_count=2,
                    last_success_at=utc(2026, 10, 4, 11, 0))
    session.add(source)
    session.flush()
    items = [
        RawItem(source_id=source.id, url=f"https://example.com/{n}",
                url_hash=hash_url(f"https://example.com/{n}"), title=f"Shooting {n}",
                body="Body text", published_at=utc(2026, 10, 4, 12, n),
                status="classified")
        for n in range(3)
    ]
    session.add_all(items)
    session.flush()
    session.add_all([
        Classification(raw_item_id=item.id, is_crime=True, crime_type="shooting",
                       was_shooting=True, in_stl=True,
                       occurred_at=utc(2026, 10, 4, 5, 40), time_precision="exact",
                       location="Grand and Gravois", neighborhood="Tower Grove East",
                       is_followup=False, confidence=0.9, model="gemini-test",
                       prompt_version="v6")
        for item in items
    ])
    # The merged incident gets the lower id, so it points forward to a row that a
    # copy in id order hasn't inserted yet.
    merged = Incident(crime_type="shooting", occurred_at=utc(2026, 10, 4, 5, 45),
                      status="merged", time_estimated=True)
    session.add(merged)
    session.flush()
    survivor = Incident(crime_type="homicide", occurred_at=utc(2026, 10, 4, 5, 40),
                        location="Grand and Gravois", neighborhood="Tower Grove East",
                        was_shooting=True, status="confirmed")
    session.add(survivor)
    session.flush()
    merged.merged_into_id = survivor.id
    session.add_all([
        IncidentItem(incident_id=survivor.id, raw_item_id=items[0].id),
        IncidentItem(incident_id=survivor.id, raw_item_id=items[1].id),
        IncidentItem(incident_id=merged.id, raw_item_id=items[2].id),
    ])
    session.add_all([
        PipelineRun(started_at=utc(2026, 10, 4, 12, 0), finished_at=utc(2026, 10, 4, 12, 1),
                    status="success", fetch_counts={"new": 3, "sources": {"KSDK": 3}},
                    classify_counts={"classified": 3}, match_counts={"created": 1}),
        PipelineRun(started_at=utc(2026, 10, 4, 12, 10), status="failed",
                    fetch_error="Traceback: boom"),
        AlertSent(key="source_down:1", last_sent_at=utc(2026, 10, 4, 12, 2)),
        JobLock(name="pipeline", owner="abc", acquired_at=utc(2026, 10, 4, 12, 10),
                expires_at=utc(2026, 10, 4, 12, 40)),
    ])
    session.commit()


def snapshot(engine, skip=("job_locks",)) -> dict[str, list[dict]]:
    """Every app table's rows (as typed Python values), in primary-key order."""
    from sqlalchemy import select

    from db import Base

    out = {}
    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name in skip:
                continue
            rows = conn.execute(select(table).order_by(*table.primary_key.columns))
            out[table.name] = [dict(r) for r in rows.mappings()]
    return out
