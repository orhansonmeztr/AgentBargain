"""initial_schema

Revision ID: 28597065d308
Revises:
Create Date: 2026-06-23 11:25:27.846661

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "28597065d308"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id    TEXT PRIMARY KEY,
            budget      REAL NOT NULL DEFAULT 0.0,
            capacity    INTEGER NOT NULL DEFAULT 1,
            reputation_score REAL NOT NULL DEFAULT 0.0
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS services (
            service_id  TEXT PRIMARY KEY,
            seller_id   TEXT NOT NULL,
            title       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            category_id TEXT NOT NULL DEFAULT '',
            category_name TEXT NOT NULL DEFAULT '',
            base_price  REAL NOT NULL DEFAULT 0.0,
            sla_terms   TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (seller_id) REFERENCES agents(agent_id)
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS negotiations (
            negotiation_id TEXT PRIMARY KEY,
            buyer_id    TEXT NOT NULL,
            seller_id   TEXT NOT NULL,
            service_id  TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'active',
            rounds      INTEGER NOT NULL DEFAULT 0,
            final_price REAL,
            final_sla   TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (buyer_id) REFERENCES agents(agent_id),
            FOREIGN KEY (seller_id) REFERENCES agents(agent_id),
            FOREIGN KEY (service_id) REFERENCES services(service_id)
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS agreements (
            agreement_id TEXT PRIMARY KEY,
            buyer_id    TEXT NOT NULL,
            seller_id   TEXT NOT NULL,
            service_id  TEXT NOT NULL,
            final_price REAL NOT NULL,
            final_sla   TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (buyer_id) REFERENCES agents(agent_id),
            FOREIGN KEY (seller_id) REFERENCES agents(agent_id),
            FOREIGN KEY (service_id) REFERENCES services(service_id)
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id   TEXT PRIMARY KEY,
            agreement_id  TEXT NOT NULL,
            evaluator_id  TEXT NOT NULL,
            target_id     TEXT NOT NULL,
            score         REAL NOT NULL,
            comments      TEXT,
            timestamp     TEXT NOT NULL,
            FOREIGN KEY (agreement_id) REFERENCES agreements(agreement_id),
            FOREIGN KEY (target_id) REFERENCES agents(agent_id)
        );
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF NOT EXISTS feedback;")
    op.execute("DROP TABLE IF NOT EXISTS agreements;")
    op.execute("DROP TABLE IF NOT EXISTS negotiations;")
    op.execute("DROP TABLE IF NOT EXISTS services;")
    op.execute("DROP TABLE IF NOT EXISTS agents;")
