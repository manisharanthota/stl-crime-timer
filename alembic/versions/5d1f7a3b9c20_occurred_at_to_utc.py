"""convert occurred_at from St. Louis local time to UTC

Before UTCDateTime, SQLite dropped the offset from LLM-provided occurred_at values,
leaving St. Louis wall-clock times. published_at, created_at, and last_success_at
were already UTC wall-clock times, so they need no data change.

Revision ID: 5d1f7a3b9c20
Revises: c62cefc9aa3b
Create Date: 2026-10-04 12:00:00

"""
from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from timeutil import LOCAL_TZ


# revision identifiers, used by Alembic.
revision: str = '5d1f7a3b9c20'
down_revision: Union[str, Sequence[str], None] = 'c62cefc9aa3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("classifications", "incidents")
_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _local_to_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc).replace(tzinfo=None)


def _utc_to_local(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ).replace(tzinfo=None)


def _convert(convert) -> None:
    bind = op.get_bind()
    # Postgres timestamptz kept the offset, so only SQLite data is wrong.
    if bind.dialect.name != "sqlite":
        return
    for table in TABLES:
        rows = bind.execute(
            sa.text(f"SELECT id, occurred_at FROM {table} WHERE occurred_at IS NOT NULL")
        ).all()
        for row_id, raw in rows:
            value = datetime.fromisoformat(raw) if isinstance(raw, str) else raw
            bind.execute(
                sa.text(f"UPDATE {table} SET occurred_at = :value WHERE id = :id"),
                {"value": convert(value).strftime(_FORMAT), "id": row_id},
            )


def upgrade() -> None:
    _convert(_local_to_utc)


def downgrade() -> None:
    _convert(_utc_to_local)
