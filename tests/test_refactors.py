"""Behaviour pinned by the refactors: strategy errors, setup errors, the
neutral reputation default and its data migration, and the shared turn helpers."""

from __future__ import annotations

import sqlite3

import pytest
from alembic import command
from alembic.config import Config

from agent_bargain.agents.guardrails import (
    GuardedDecision,
    opponent_reputation,
    turn_update,
)
from agent_bargain.config import PROJECT_ROOT
from agent_bargain.engine.errors import (
    NegotiationSetupError,
    OpeningBelowFloorError,
    SellerMismatchError,
    ServiceNotRegisteredError,
)
from agent_bargain.engine.negotiation_runner import prepare_negotiation
from agent_bargain.engine.reputation import NEUTRAL_REPUTATION_SCORE
from agent_bargain.models.negotiation import NegotiationRequest
from agent_bargain.models.service import ServiceListing
from agent_bargain.storage.database import DatabaseManager
from agent_bargain.strategies import StrategyRegistry, UnknownStrategyError

# -- strategies --------------------------------------------------------------


def test_unknown_strategy_error_is_readable_and_still_a_keyerror():
    with pytest.raises(UnknownStrategyError) as info:
        StrategyRegistry.get("nope")
    assert isinstance(info.value, KeyError)
    assert str(info.value).startswith("Strategy 'nope' not found. Available: [")
    assert "cooperative" in info.value.available


def test_request_reports_unknown_strategy_without_keyerror_quotes():
    with pytest.raises(ValueError, match=r"Strategy 'nope' not found"):
        NegotiationRequest(
            buyer_id="b",
            seller_id="s",
            service_id="x",
            max_budget=1.0,
            engine="strategy",
            buyer_strategy="nope",
            seller_strategy="cooperative",
        )


# -- setup errors are transport-neutral -------------------------------------


@pytest.fixture
async def db(tmp_path):
    manager = DatabaseManager(tmp_path / "refactor.db")
    await manager.connect()
    await manager.ensure_agent("seller_1")
    await manager.upsert_service(
        ServiceListing(
            service_id="srv_1",
            seller_id="seller_1",
            title="Test",
            base_price=400.0,
            floor_price=100.0,
            sla_terms="Standard",
        )
    )
    yield manager
    await manager.close()


@pytest.mark.parametrize(
    ("overrides", "error", "status"),
    [
        ({"service_id": "missing"}, ServiceNotRegisteredError, 404),
        ({"seller_id": "someone_else"}, SellerMismatchError, 409),
        ({"initial_price": 50.0}, OpeningBelowFloorError, 422),
    ],
)
async def test_prepare_raises_domain_errors(db, overrides, error, status):
    request = NegotiationRequest(
        **{
            "buyer_id": "buyer_1",
            "seller_id": "seller_1",
            "service_id": "srv_1",
            "max_budget": 500.0,
            **overrides,
        }
    )
    with pytest.raises(error) as info:
        await prepare_negotiation(db, request)
    assert isinstance(info.value, NegotiationSetupError)
    assert info.value.status_code == status
    assert info.value.detail


# -- neutral reputation for new agents --------------------------------------


async def test_new_agents_start_at_the_neutral_score(db):
    await db.ensure_agent("fresh")
    agents = {a["agent_id"]: a for a in await db.list_agents()}
    assert agents["fresh"]["reputation_score"] == NEUTRAL_REPUTATION_SCORE


def test_migration_neutralises_placeholder_scores_only(tmp_path, monkeypatch):
    from agent_bargain.config import settings

    db_file = tmp_path / "migrate.db"
    monkeypatch.setattr(settings, "sqlite_db_path", str(db_file))
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    command.upgrade(cfg, "c3d4e5f6a7b8")

    conn = sqlite3.connect(db_file)
    conn.executemany(
        "INSERT INTO agents (agent_id, budget, capacity, reputation_score) VALUES (?, 0, 1, ?)",
        [("placeholder", 0.0), ("rated_low", 0.0), ("rated", 4.2)],
    )
    conn.execute(
        "INSERT INTO feedback (feedback_id, agreement_id, evaluator_id, target_id, score, "
        "timestamp) VALUES ('f1', 'a1', 'e', 'rated_low', 1.0, '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    command.upgrade(cfg, "head")

    conn = sqlite3.connect(db_file)
    scores = dict(conn.execute("SELECT agent_id, reputation_score FROM agents"))
    conn.close()
    assert scores == {
        "placeholder": NEUTRAL_REPUTATION_SCORE,
        "rated_low": 0.0,  # has feedback: left for the reputation engine
        "rated": 4.2,
    }


# -- shared turn helpers -----------------------------------------------------


def test_opponent_reputation_reads_the_other_side():
    state = {
        "buyer_reputation_score": 1.5,
        "buyer_reputation_summary": "buyer",
        "seller_reputation_score": 4.5,
        "seller_reputation_summary": "seller",
    }
    assert opponent_reputation(state, "buyer") == (4.5, "seller")
    assert opponent_reputation(state, "seller") == (1.5, "buyer")
    assert opponent_reputation({}, "buyer")[0] == NEUTRAL_REPUTATION_SCORE


def test_turn_update_carries_the_applied_decision():
    update = turn_update("seller", GuardedDecision("counter", 250.0, "Gold", "why"))
    assert update["current_price"] == 250.0
    assert update["last_action"] == "counter"
    assert update["last_actor"] == "seller"
    raw = update["messages"][0].additional_kwargs["raw_data"]
    assert raw == {"action": "counter", "price": 250.0, "sla": "Gold", "reasoning": "why"}
