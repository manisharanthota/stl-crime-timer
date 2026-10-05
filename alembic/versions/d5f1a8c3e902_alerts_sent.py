"""alerts_sent

Revision ID: d5f1a8c3e902
Revises: c9a4e2d7f813
Create Date: 2026-10-05 23:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f1a8c3e902'
down_revision: Union[str, Sequence[str], None] = 'c9a4e2d7f813'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('alerts_sent',
    sa.Column('key', sa.String(length=200), nullable=False),
    sa.Column('last_sent_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('key')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('alerts_sent')
