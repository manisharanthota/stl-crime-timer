"""add incidents.manual_occurred_at / manual_location / manual_neighborhood

Revision ID: f2b7c4e1a905
Revises: d5f1a8c3e902
Create Date: 2026-10-07 05:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2b7c4e1a905'
down_revision: Union[str, Sequence[str], None] = 'd5f1a8c3e902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

COLUMNS = ("manual_occurred_at", "manual_location", "manual_neighborhood")


def upgrade() -> None:
    with op.batch_alter_table("incidents") as batch_op:
        for name in COLUMNS:
            batch_op.add_column(
                sa.Column(name, sa.Boolean(), server_default="0", nullable=False)
            )


def downgrade() -> None:
    # Native DROP COLUMN (SQLite >= 3.35): a batch table rebuild of incidents would
    # fail on the incident_items foreign key, since db.py enables FKs on SQLite.
    for name in reversed(COLUMNS):
        op.drop_column("incidents", name)
