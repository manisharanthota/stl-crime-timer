"""add classifications.time_precision

Only "exact" occurred_at values count as reported times, so a guessed midnight never
beats a stated time. Backfill for rows from before the field existed: no
occurred_at -> unknown; exactly 00:00 St. Louis time -> date_only (what the model
puts when an article gives only a date); anything else -> exact.

Revision ID: c9a4e2d7f813
Revises: b3d8f1c47e26
Create Date: 2026-10-05 20:00:00

"""
from datetime import time, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from timeutil import to_local


# revision identifiers, used by Alembic.
revision: str = 'c9a4e2d7f813'
down_revision: Union[str, Sequence[str], None] = 'b3d8f1c47e26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

classifications = sa.table(
    "classifications",
    sa.column("id", sa.Integer),
    sa.column("occurred_at", sa.DateTime(timezone=True)),
    sa.column("time_precision", sa.String),
)


def _precision(occurred_at) -> str:
    if occurred_at is None:
        return "unknown"
    if occurred_at.tzinfo is None:  # SQLite stores naive UTC
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    return "date_only" if to_local(occurred_at).time() == time(0, 0) else "exact"


def upgrade() -> None:
    with op.batch_alter_table("classifications") as batch_op:
        batch_op.add_column(sa.Column(
            "time_precision",
            sa.Enum(
                "exact", "date_only", "unknown",
                name="time_precision", native_enum=False, create_constraint=True,
            ),
            server_default="exact",
            nullable=False,
        ))

    conn = op.get_bind()
    rows = conn.execute(sa.select(classifications.c.id, classifications.c.occurred_at)).all()
    for precision in ("unknown", "date_only"):
        ids = [r.id for r in rows if _precision(r.occurred_at) == precision]
        if ids:
            conn.execute(
                classifications.update()
                .where(classifications.c.id.in_(ids))
                .values(time_precision=precision)
            )


def downgrade() -> None:
    with op.batch_alter_table("classifications") as batch_op:
        # The CHECK constraint names the column, so it must go first.
        batch_op.drop_constraint("time_precision", type_="check")
        batch_op.drop_column("time_precision")
