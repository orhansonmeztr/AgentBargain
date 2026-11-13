"""add_evaluator_model

Revision ID: dc0130fd8508
Revises: 28597065d308
Create Date: 2026-06-23 11:26:00.903214

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dc0130fd8508"
down_revision: str | Sequence[str] | None = "28597065d308"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE feedback ADD COLUMN evaluator_model TEXT;")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER TABLE feedback DROP COLUMN evaluator_model;")
