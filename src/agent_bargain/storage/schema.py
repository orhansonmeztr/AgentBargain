"""Authoritative SQLite DDL for AgentBargain.

``DatabaseManager._init_schema`` executes :data:`SCHEMA_SQL` so a fresh app
start produces a usable database without requiring ``alembic upgrade head``.

The Alembic migration chain must converge on exactly this shape. That is not
enforced by convention: ``tests/test_schema_drift.py`` builds one database from
the migrations and one from ``SCHEMA_SQL`` and fails if the resulting columns
differ in name, type, nullability or default.
"""

from __future__ import annotations

#: Full schema, idempotent. Every table uses ``IF NOT EXISTS`` so this is safe
#: to run against an already-initialized database.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id         TEXT PRIMARY KEY,
    budget           REAL NOT NULL DEFAULT 0.0,
    capacity         INTEGER NOT NULL DEFAULT 1,
    reputation_score REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS services (
    service_id    TEXT PRIMARY KEY,
    seller_id     TEXT NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    category_id   TEXT NOT NULL DEFAULT '',
    category_name TEXT NOT NULL DEFAULT '',
    base_price    REAL NOT NULL DEFAULT 0.0,
    floor_price   REAL NOT NULL DEFAULT 0.0,
    sla_terms     TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (seller_id) REFERENCES agents(agent_id)
);

CREATE TABLE IF NOT EXISTS negotiations (
    negotiation_id TEXT PRIMARY KEY,
    buyer_id       TEXT NOT NULL,
    seller_id      TEXT NOT NULL,
    service_id     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'active',
    rounds         INTEGER NOT NULL DEFAULT 0,
    max_rounds     INTEGER,
    final_price    REAL,
    final_sla      TEXT,
    max_budget     REAL,
    min_price      REAL,
    initial_price  REAL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    FOREIGN KEY (buyer_id)  REFERENCES agents(agent_id),
    FOREIGN KEY (seller_id) REFERENCES agents(agent_id),
    FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS negotiation_messages (
    message_id      TEXT PRIMARY KEY,
    negotiation_id  TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    raw_data        TEXT,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (negotiation_id) REFERENCES negotiations(negotiation_id)
);

CREATE TABLE IF NOT EXISTS agreements (
    agreement_id TEXT PRIMARY KEY,
    buyer_id     TEXT NOT NULL,
    seller_id    TEXT NOT NULL,
    service_id   TEXT NOT NULL,
    final_price  REAL NOT NULL,
    final_sla    TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    FOREIGN KEY (buyer_id)  REFERENCES agents(agent_id),
    FOREIGN KEY (seller_id) REFERENCES agents(agent_id),
    FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     TEXT PRIMARY KEY,
    agreement_id    TEXT NOT NULL,
    evaluator_id    TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    score           REAL NOT NULL,
    comments        TEXT,
    timestamp       TEXT NOT NULL,
    evaluator_model TEXT,
    FOREIGN KEY (agreement_id) REFERENCES agreements(agreement_id),
    FOREIGN KEY (target_id) REFERENCES agents(agent_id)
);

-- Every hot query filters or sorts on one of these, and none of them is a
-- primary key, so without an index each is a full table scan that grows with
-- the marketplace's entire history:
--   feedback.target_id                  -> reputation scoring, per negotiation
--   negotiation_messages.negotiation_id -> message + trajectory loading
--   negotiations.created_at             -> ORDER BY on history/aggregate calls
CREATE INDEX IF NOT EXISTS idx_feedback_target ON feedback(target_id);
CREATE INDEX IF NOT EXISTS idx_messages_negotiation ON negotiation_messages(negotiation_id);
CREATE INDEX IF NOT EXISTS idx_negotiations_created ON negotiations(created_at);
"""
