"""add was_shooting to classifications and incidents

A fatal shooting is stored as crime_type=homicide; was_shooting lets it count toward
the shooting timer too. Backfill: classifications with crime_type=shooting, and
incidents that are shootings or have a linked item whose newest classification was a
shooting (e.g. merged shooting -> homicide upgrades). Older homicide classifications
can't tell whether a gun was involved, so they stay false.

Revision ID: a7c3e9f2b514
Revises: e4d4a3446628
Create Date: 2026-10-05 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7c3e9f2b514'
down_revision: Union[str, Sequence[str], None] = 'e4d4a3446628'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("classifications", "incidents")

classifications = sa.table(
    "classifications",
    sa.column("id", sa.Integer),
    sa.column("raw_item_id", sa.Integer),
    sa.column("crime_type", sa.String),
    sa.column("was_shooting", sa.Boolean),
)
incidents = sa.table(
    "incidents",
    sa.column("id", sa.Integer),
    sa.column("crime_type", sa.String),
    sa.column("was_shooting", sa.Boolean),
)
incident_items = sa.table(
    "incident_items",
    sa.column("incident_id", sa.Integer),
    sa.column("raw_item_id", sa.Integer),
)


def upgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(
                sa.Column("was_shooting", sa.Boolean(), server_default="0", nullable=False)
            )

    op.execute(
        classifications.update()
        .where(classifications.c.crime_type == "shooting")
        .values(was_shooting=True)
    )

    latest = (
        sa.select(sa.func.max(classifications.c.id))
        .group_by(classifications.c.raw_item_id)
        .scalar_subquery()
    )
    shot = (
        sa.select(incident_items.c.incident_id)
        .join(
            classifications,
            classifications.c.raw_item_id == incident_items.c.raw_item_id,
        )
        .where(classifications.c.id.in_(latest), classifications.c.was_shooting.is_(True))
    )
    op.execute(
        incidents.update()
        .where(sa.or_(incidents.c.crime_type == "shooting", incidents.c.id.in_(shot)))
        .values(was_shooting=True)
    )


def downgrade() -> None:
    # Native DROP COLUMN (SQLite >= 3.35): a batch table rebuild of incidents would
    # fail on the incident_items foreign key, since db.py enables FKs on SQLite.
    for table in TABLES:
        op.drop_column(table, "was_shooting")
