"""add neighborhood, is_followup, merged incidents

classifications: neighborhood, is_followup (prompt v4).
incidents: neighborhood, merged_into_id, and status 'merged'. Changing the status
CHECK constraint rebuilds incidents on SQLite (see the foreign_keys handling in
env.py).

Revision ID: b3d8f1c47e26
Revises: a7c3e9f2b514
Create Date: 2026-10-05 18:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3d8f1c47e26'
down_revision: Union[str, Sequence[str], None] = 'a7c3e9f2b514'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_opts = {"name": "incident_status", "native_enum": False, "create_constraint": True}
OLD_STATUS = sa.Enum("confirmed", "review", "rejected", **_opts)
NEW_STATUS = sa.Enum("confirmed", "review", "rejected", "merged", **_opts)

incidents = sa.table(
    "incidents",
    sa.column("status", sa.String),
)


def upgrade() -> None:
    with op.batch_alter_table("classifications") as batch_op:
        batch_op.add_column(sa.Column("neighborhood", sa.String(length=100), nullable=True))
        batch_op.add_column(
            sa.Column("is_followup", sa.Boolean(), server_default="0", nullable=False)
        )

    with op.batch_alter_table("incidents") as batch_op:
        batch_op.add_column(sa.Column("neighborhood", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("merged_into_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_incidents_merged_into_id", "incidents", ["merged_into_id"], ["id"]
        )
        batch_op.alter_column(
            "status",
            existing_type=OLD_STATUS,
            type_=NEW_STATUS,
            existing_nullable=False,
            existing_server_default="review",
        )


def downgrade() -> None:
    # 'merged' doesn't exist before this revision; put those back up for review.
    op.execute(incidents.update().where(incidents.c.status == "merged").values(status="review"))
    with op.batch_alter_table("incidents") as batch_op:
        batch_op.alter_column(
            "status",
            existing_type=NEW_STATUS,
            type_=OLD_STATUS,
            existing_nullable=False,
            existing_server_default="review",
        )
        batch_op.drop_constraint("fk_incidents_merged_into_id", type_="foreignkey")
        batch_op.drop_column("merged_into_id")
        batch_op.drop_column("neighborhood")

    for column in ("is_followup", "neighborhood"):
        op.drop_column("classifications", column)
