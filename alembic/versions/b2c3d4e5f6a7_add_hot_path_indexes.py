"""add_hot_path_indexes

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-11-03 13:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Indexes for the columns the hot queries filter and sort on. None is a
#: primary key, so without these every reputation lookup, message load and
#: history listing was a full table scan growing with the whole marketplace.
_INDEXES = [
    ("idx_feedback_target", "feedback", "target_id"),
    ("idx_messages_negotiation", "negotiation_messages", "negotiation_id"),
    ("idx_negotiations_created", "negotiations", "created_at"),
]


def upgrade() -> None:
    """Upgrade schema."""
    for name, table, column in _INDEXES:
        op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column});")


def downgrade() -> None:
    """Downgrade schema."""
    for name, _, _ in reversed(_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {name};")
