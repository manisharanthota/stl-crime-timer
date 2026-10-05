"""add incidents.time_estimated

Revision ID: 8e4b2c6d1a70
Revises: 5d1f7a3b9c20
Create Date: 2026-10-04 18:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8e4b2c6d1a70'
down_revision: Union[str, Sequence[str], None] = '5d1f7a3b9c20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("incidents") as batch_op:
        batch_op.add_column(
            sa.Column("time_estimated", sa.Boolean(), server_default="0", nullable=False)
        )


def downgrade() -> None:
    # Native DROP COLUMN (SQLite >= 3.35): a batch table rebuild of incidents would
    # fail on the incident_items foreign key, since db.py enables FKs on SQLite.
    op.drop_column("incidents", "time_estimated")
