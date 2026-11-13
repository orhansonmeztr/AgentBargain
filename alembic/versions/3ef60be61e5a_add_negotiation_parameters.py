"""add_negotiation_parameters

Revision ID: 3ef60be61e5a
Revises: 57abd7992f7e
Create Date: 2026-06-28 12:09:20.772260

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3ef60be61e5a"
down_revision: str | Sequence[str] | None = "57abd7992f7e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE negotiations ADD COLUMN max_budget REAL;")
    op.execute("ALTER TABLE negotiations ADD COLUMN min_price REAL;")
    op.execute("ALTER TABLE negotiations ADD COLUMN initial_price REAL;")


def downgrade() -> None:
    """Downgrade schema."""
    # SQLite does not support dropping columns without recreating the table.
    # Downgrading this migration is intentionally unsupported.
    raise NotImplementedError(
        "Downgrade of 3ef60be61e5a is not supported: SQLite cannot drop columns. "
        "Restore from a backup or recreate the database."
    )
