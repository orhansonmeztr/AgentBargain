"""add_negotiation_history

Revision ID: 57abd7992f7e
Revises: dc0130fd8508
Create Date: 2026-06-28 11:52:26.062597

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "57abd7992f7e"
down_revision: str | Sequence[str] | None = "dc0130fd8508"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        CREATE TABLE IF NOT EXISTS negotiation_messages (
            message_id TEXT PRIMARY KEY,
            negotiation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            raw_data TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (negotiation_id) REFERENCES negotiations(negotiation_id)
        );
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF NOT EXISTS negotiation_messages;")
