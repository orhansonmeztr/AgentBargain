"""add_negotiation_max_rounds

Revision ID: a1b2c3d4e5f6
Revises: 3ef60be61e5a
Create Date: 2026-11-03 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "3ef60be61e5a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema.

    ``max_rounds`` records the round limit a negotiation actually ran with.
    The state machine has always supported a per-negotiation limit, but it was
    never persisted, so game-theory analytics had to assume the global
    ``MAX_NEGOTIATION_ROUNDS`` when computing round efficiency.

    Nullable: rows written before this migration have no recorded limit and
    analytics falls back to the global setting for them.
    """
    op.execute("ALTER TABLE negotiations ADD COLUMN max_rounds INTEGER;")


def downgrade() -> None:
    """Downgrade schema."""
    # SQLite cannot drop columns without recreating the table.
    raise NotImplementedError(
        "Downgrade of a1b2c3d4e5f6 is not supported: SQLite cannot drop columns. "
        "Restore from a backup or recreate the database."
    )
