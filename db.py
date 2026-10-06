"""SQLAlchemy engine, session factory, and declarative base."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Engine, TypeDecorator, create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import get_settings
from timeutil import to_utc


class Base(DeclarativeBase):
    pass


class UTCDateTime(TypeDecorator):
    """DateTime that is always UTC in the database and aware-UTC in Python.

    Values are converted with timeutil.to_utc on the way in (naive = America/Chicago).
    SQLite can't store an offset, so it gets naive UTC; Postgres gets timestamptz.
    Values read back are aware UTC on both.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        value = to_utc(value)
        if dialect.name == "sqlite":
            return value.replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    # SQLite ignores foreign keys unless enabled per connection.
    if type(dbapi_connection).__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def make_engine(url: str, **kwargs) -> Engine:
    """Engine for any supported URL. Postgres connections are checked before use:
    hosted Postgres (Supabase) closes idle ones."""
    if not url.startswith("sqlite"):
        kwargs.setdefault("pool_pre_ping", True)
    return create_engine(url, **kwargs)


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
