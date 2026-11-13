"""Regression tests for the review fixes: limit leaks, NaN prices, atomic
persistence, final terms on unsuccessful outcomes, aggregate counts and
evaluation timeouts."""

from __future__ import annotations

import asyncio
import math

import pytest

import agent_bargain.agents.evaluator as evaluator_module
import agent_bargain.engine.negotiation_runner as runner_module
from agent_bargain.agents.guardrails import (
    GuardedDecision,
    apply_guardrails,
    build_turn_message,
    history_for,
)
from agent_bargain.engine.negotiation import create_initial_state, create_strategy_graph
from agent_bargain.engine.negotiation_runner import prepare_negotiation, run_negotiation
from agent_bargain.llm.schemas import NegotiationDecision
from agent_bargain.models.negotiation import NegotiationRequest
from agent_bargain.models.service import ServiceListing
from agent_bargain.storage.database import DatabaseManager
from agent_bargain.strategies.base import NegotiationContext
from tests.conftest import register_service, start_payload

_REQUEST = NegotiationRequest(
    buyer_id="buyer_1", seller_id="seller_1", service_id="srv_1", max_budget=500.0
)


@pytest.fixture
async def db(tmp_path):
    manager = DatabaseManager(tmp_path / "fixes.db")
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


# -- 1. no private limit reaches the opponent -------------------------------


@pytest.mark.parametrize(
    ("role", "action", "price", "current", "limit"),
    [
        ("buyer", "counter", 900.0, 950.0, 487.0),
        ("seller", "counter", 50.0, 30.0, 137.0),
        ("buyer", "accept", 0.0, 900.0, 487.0),
        ("seller", "accept", 0.0, 50.0, 137.0),
    ],
)
def test_guardrail_notes_never_contain_the_limit(role, action, price, current, limit):
    g = apply_guardrails(
        action=action,
        price=price,
        sla="S",
        reasoning="",
        role=role,
        limit=limit,
        current_price=current,
    )
    assert "Guardrail" in g.reasoning
    assert str(int(limit)) not in g.reasoning


def test_opponent_history_omits_reasoning_but_own_keeps_it():
    buyer_msg = build_turn_message(
        role="buyer",
        decision=GuardedDecision("counter", 200.0, "S", "my budget is $487"),
    )
    seller_msg = build_turn_message(
        role="seller",
        decision=GuardedDecision("counter", 300.0, "S", "my floor is $137"),
    )
    seen_by_seller = history_for("seller", [buyer_msg, seller_msg])
    assert "487" not in "\n".join(seen_by_seller)
    assert "my floor is $137" in seen_by_seller[1]
    assert seen_by_seller[0] == "BUYER: COUNTER - Price: $200.0, SLA: S."


def test_strategy_graph_never_shows_the_opponents_limit_to_strategies(monkeypatch):
    seen: list[str] = []
    import agent_bargain.agents.strategy_node as node_module

    real = node_module.NegotiationContext

    def _spy(**kwargs):
        if kwargs["role"] == "seller":
            seen.extend(kwargs["message_history"])
        return real(**kwargs)

    monkeypatch.setattr(node_module, "NegotiationContext", _spy)
    state = create_initial_state(
        service_id="s",
        service_title="t",
        buyer_id="b",
        seller_id="s",
        max_budget=487.0,
        min_price=137.0,
        initial_price=400.0,
        engine="strategy",
        buyer_strategy="aggressive",
        seller_strategy="aggressive",
    )
    create_strategy_graph("aggressive", "aggressive").invoke(state)
    assert seen and all("487" not in line for line in seen)


# -- 2. non-finite prices ----------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_counter_is_rejected(bad):
    g = apply_guardrails(
        action="counter",
        price=bad,
        sla="S",
        reasoning="",
        role="seller",
        limit=100.0,
        current_price=50.0,
    )
    assert g.action == "reject"
    assert math.isfinite(g.price)


def test_accept_against_a_non_finite_table_price_is_rejected():
    g = apply_guardrails(
        action="accept",
        price=0.0,
        sla="S",
        reasoning="",
        role="buyer",
        limit=500.0,
        current_price=math.nan,
    )
    assert g.action == "reject"


def test_llm_schema_rejects_nan():
    with pytest.raises(ValueError):
        NegotiationDecision(action="counter", price=math.nan, sla="x", reasoning="y")


def test_price_history_skips_malformed_entries():
    ctx = NegotiationContext(
        role="buyer",
        max_budget=1,
        min_price=0,
        current_price=1,
        current_sla="S",
        initial_price=1,
        round=0,
        max_rounds=5,
        opponent_reputation=3,
        opponent_reputation_summary="",
        metadata={"price_history": [("seller", "abc"), ("seller", math.nan), ("seller", 5)]},
    )
    assert ctx.price_history == [("seller", 5.0)]


# -- 3. floor above asking price --------------------------------------------


def test_registration_rejects_floor_above_base_price(client):
    response = client.post(
        "/api/v1/marketplace/services",
        json={
            "service_id": "x",
            "seller_id": "s",
            "title": "t",
            "base_price": 100.0,
            "floor_price": 150.0,
        },
    )
    assert response.status_code == 422


# -- 4. final terms only for accepted deals -----------------------------------


async def test_rejected_negotiation_has_no_final_price(db, monkeypatch):
    class _RejectGraph:
        async def astream(self, state):
            yield {"buyer": {"current_price": 333.0, "last_action": "reject"}}
            yield {"rejected": {"status": "rejected"}}

    monkeypatch.setattr(runner_module, "create_negotiation_graph", lambda: _RejectGraph())
    neg_id, initial = await prepare_negotiation(db, _REQUEST)
    summary = await run_negotiation(initial, neg_id, db)

    assert summary["status"] == "rejected"
    assert summary["final_price"] is None
    row = await db.get_negotiation(neg_id)
    assert row["final_price"] is None and row["final_sla"] is None


# -- 8. atomic outcome -------------------------------------------------------


async def test_failed_transcript_write_rolls_back_the_whole_outcome(db, monkeypatch):
    class _AcceptGraph:
        async def astream(self, state):
            yield {"buyer": {"last_action": "accept"}}
            yield {"accepted": {"status": "accepted"}}

    async def _boom(conn, messages):
        raise RuntimeError("disk full")

    import agent_bargain.storage.database as db_module

    monkeypatch.setattr(runner_module, "create_negotiation_graph", lambda: _AcceptGraph())
    monkeypatch.setattr(db_module, "_write_messages", _boom)

    neg_id, initial = await prepare_negotiation(db, _REQUEST)
    summary = await run_negotiation(initial, neg_id, db)

    assert summary["status"] == "failed"
    assert (await db.get_negotiation(neg_id))["status"] == "failed"
    cursor = await db.conn.execute("SELECT COUNT(*) FROM agreements")
    assert (await cursor.fetchone())[0] == 0, "no orphan agreement for a failed outcome"


# -- 9. graph construction failure ------------------------------------------


async def test_graph_build_failure_is_recorded_as_failed(db, monkeypatch):
    def _broken():
        raise RuntimeError("cannot build graph")

    monkeypatch.setattr(runner_module, "create_negotiation_graph", _broken)
    neg_id, initial = await prepare_negotiation(db, _REQUEST)
    summary = await run_negotiation(initial, neg_id, db)
    assert summary["status"] == "failed"
    assert (await db.get_negotiation(neg_id))["status"] == "failed"


# -- 7. evaluation timeout ---------------------------------------------------


async def test_evaluation_times_out_to_none(monkeypatch):
    class _Hanging:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt):
            await asyncio.Event().wait()

    monkeypatch.setattr(evaluator_module, "get_evaluation_llm", lambda *a, **k: _Hanging())
    monkeypatch.setattr(evaluator_module.settings, "evaluation_timeout_seconds", 0.05)
    result = await evaluator_module.evaluate_agreement(final_price=1.0, final_sla="S", history=[])
    assert result is None


# -- 5. aggregate counts every row ------------------------------------------


async def test_negotiation_stats_counts_every_row(db):
    from agent_bargain.models.negotiation import NegotiationRecord

    await db.ensure_agent("buyer_1")
    for i, status in enumerate(["accepted", "accepted", "rejected", "timeout", "failed"]):
        await db.save_negotiation(
            NegotiationRecord(
                negotiation_id=f"n{i}",
                buyer_id="buyer_1",
                seller_id="seller_1",
                service_id="srv_1",
                status=status,
                rounds=2,
                final_price=200.0 if status == "accepted" else None,
            )
        )
    stats = await db.negotiation_stats()
    assert stats["total"] == 5
    assert (stats["accepted"], stats["rejected"], stats["timeout"]) == (2, 1, 1)
    assert stats["avg_final_price"] == 200.0
    assert len(await db.list_negotiations(status="accepted")) == 2


# -- 10. small fixes ---------------------------------------------------------


async def test_upsert_refuses_another_sellers_listing(db):
    await db.ensure_agent("intruder")
    taken = await db.upsert_service(
        ServiceListing(
            service_id="srv_1",
            seller_id="intruder",
            title="Mine now",
            base_price=1.0,
            sla_terms="S",
        )
    )
    assert taken is False
    service = await db.get_service("srv_1")
    assert service.seller_id == "seller_1" and service.title == "Test"


async def test_messages_with_equal_timestamps_keep_insertion_order(db):
    from datetime import UTC, datetime

    from agent_bargain.models.negotiation import NegotiationMessage

    neg_id, _ = await prepare_negotiation(db, _REQUEST)
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    # Inserted in turn order with identical timestamps; ids sort the other way.
    msgs = [
        NegotiationMessage(
            message_id=f"m{9 - i}",
            negotiation_id=neg_id,
            role="buyer",
            content=str(i),
            timestamp=ts,
        )
        for i in range(5)
    ]
    await db.save_negotiation_messages(msgs)
    rows = await db.get_messages_for_negotiation(neg_id)
    assert [r["content"] for r in rows] == ["0", "1", "2", "3", "4"]


def test_messages_for_unknown_negotiation_is_404(client):
    assert client.get("/api/v1/negotiations/neg_nope/messages").status_code == 404


def test_messages_for_known_negotiation_is_200(client):
    register_service(client)
    neg_id = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy", buyer_strategy="cooperative", seller_strategy="cooperative"
        ),
    ).json()["negotiation_id"]
    assert client.get(f"/api/v1/negotiations/{neg_id}/messages").status_code == 200


async def test_cancelled_negotiation_is_broadcast_as_failed(db, monkeypatch):
    """Clients learn about a cancellation too, not only the database."""
    started = asyncio.Event()
    events: list[dict] = []

    class _HangingGraph:
        async def astream(self, state):
            started.set()
            await asyncio.Event().wait()
            yield {}  # pragma: no cover

    async def _capture(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(runner_module, "create_negotiation_graph", lambda: _HangingGraph())
    neg_id, initial = await prepare_negotiation(db, _REQUEST)
    task = asyncio.create_task(run_negotiation(initial, neg_id, db, broadcast=_capture))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == [
        {
            "type": "negotiation_update",
            "negotiation_id": neg_id,
            "status": "failed",
            "rounds": 0,
            "final_price": None,
        }
    ]
