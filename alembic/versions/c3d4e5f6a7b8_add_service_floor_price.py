"""add_service_floor_price

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-11-03 14:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema.

    ``floor_price`` is the seller's walk-away point. It moves onto the service
    listing so a negotiating counterparty can no longer dictate it through
    ``POST /negotiations/start`` — previously any caller could send
    ``min_price=0`` and win every deal, which made the game-theory analytics
    measure a game the seller was never playing.

    Defaults to 0.0 for rows registered before this migration; re-register the
    service to set a real floor.
    """
    op.execute("ALTER TABLE services ADD COLUMN floor_price REAL NOT NULL DEFAULT 0.0;")


def downgrade() -> None:
    """Downgrade schema."""
    # SQLite cannot drop columns without recreating the table.
    raise NotImplementedError(
        "Downgrade of c3d4e5f6a7b8 is not supported: SQLite cannot drop columns. "
        "Restore from a backup or recreate the database."
    )
