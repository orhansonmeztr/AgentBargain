"""neutral_reputation_for_new_agents

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-11-12 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Mirrors ``engine.reputation.NEUTRAL_REPUTATION_SCORE``. Frozen here on
#: purpose: a migration must keep meaning what it meant when it was written.
NEUTRAL_REPUTATION_SCORE = 3.0


def upgrade() -> None:
    """Data-only: agents with no feedback get the neutral score instead of 0.0.

    Auto-registered agents used to be inserted with ``reputation_score=0.0``,
    which the leaderboard showed as the worst possible rating although the
    agent simply had no history — the same mistake ``score_or_neutral`` exists
    to prevent in negotiation prompts. Agents that do have feedback keep their
    computed score. The column's DDL default is left alone: every insert path
    now writes the score explicitly, and changing a default in SQLite means
    rebuilding a table other tables reference.
    """
    op.execute(
        f"UPDATE agents SET reputation_score = {NEUTRAL_REPUTATION_SCORE} "
        "WHERE reputation_score = 0.0 "
        "AND agent_id NOT IN (SELECT DISTINCT target_id FROM feedback)"
    )


def downgrade() -> None:
    """Restore the old placeholder for agents that still have no feedback."""
    op.execute(
        f"UPDATE agents SET reputation_score = 0.0 "
        f"WHERE reputation_score = {NEUTRAL_REPUTATION_SCORE} "
        "AND agent_id NOT IN (SELECT DISTINCT target_id FROM feedback)"
    )
