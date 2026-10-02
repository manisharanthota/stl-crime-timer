"""Load sources from sources.yaml into the sources table, skipping known URLs."""

from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Source

DEFAULT_SOURCES_PATH = Path(__file__).parent / "sources.yaml"


def load_sources(path: str | Path = DEFAULT_SOURCES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("sources") or []


def seed_sources(session: Session, path: str | Path = DEFAULT_SOURCES_PATH) -> int:
    """Insert sources whose url isn't already stored. Returns the number inserted."""
    existing = set(session.scalars(select(Source.url)))
    inserted = 0
    for entry in load_sources(path):
        if entry["url"] in existing:
            continue
        session.add(Source(name=entry["name"], url=entry["url"], type=entry["type"]))
        existing.add(entry["url"])
        inserted += 1
    session.commit()
    return inserted


if __name__ == "__main__":
    from db import SessionLocal

    with SessionLocal() as session:
        print(f"Inserted {seed_sources(session)} source(s)")
