"""Postgres test database, opt-in through DATABASE_URL_TEST (environment or .env).

Never point it at a database you care about. Tests only touch schemas they create
themselves (pytest_<random>) and drop them afterwards, and they refuse to run against
DATABASE_URL, but use a throwaway database anyway (local Postgres or the CI service).
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import Engine, make_url, text

from config import normalize_database_url
from db import make_engine

ROOT = Path(__file__).resolve().parent.parent


def _from_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        value = dotenv_values(ROOT / ".env").get(name)
    return normalize_database_url(value) if value else None


PG_URL = _from_env("DATABASE_URL_TEST")


def check_safe() -> None:
    if PG_URL and PG_URL == _from_env("DATABASE_URL"):
        pytest.exit("DATABASE_URL_TEST must not be the same as DATABASE_URL", returncode=2)


requires_pg = pytest.mark.skipif(not PG_URL, reason="DATABASE_URL_TEST not set")


def with_search_path(url: str, schema: str) -> str:
    """URL whose connections default to `schema` (also for alembic, which only gets a
    URL)."""
    return make_url(url).update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)


@contextmanager
def temp_schema() -> Iterator[str]:
    """A fresh empty schema; yields a URL that uses it, drops it afterwards."""
    schema = f"pytest_{uuid.uuid4().hex[:12]}"
    admin = make_engine(PG_URL)
    try:
        with admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        yield with_search_path(PG_URL, schema)
    finally:
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


def truncate_all(engine: Engine, table_names: list[str]) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            f"TRUNCATE {', '.join(table_names)} RESTART IDENTITY CASCADE"
        ))
