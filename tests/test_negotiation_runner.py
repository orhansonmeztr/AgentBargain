"""``run_negotiation`` must always leave a terminal status behind.

An ``active`` row is indistinguishable from a live negotiation, so every way
out of the runner — success, crash, cancellation — has to overwrite it.
"""

from __future__ import annotations

import asyncio

import pytest

import agent_bargain.engine.negotiation_runner as runner_module
from agent_bargain.engine.negotiation_runner import prepare_negotiation, run_negotiation
from agent_bargain.models.negotiation import NegotiationRequest
from agent_bargain.models.service import ServiceListing
from agent_bargain.storage.database import DatabaseManager

_REQUEST = NegotiationRequest(
    buyer_id="buyer_1", seller_id="seller_1", service_id="srv_1", max_budget=500.0
)


@pytest.fixture
async def db(tmp_path):
    manager = DatabaseManager(tmp_path / "runner.db")
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


async def test_cancelled_negotiation_is_recorded_as_failed(db, monkeypatch):
    started = asyncio.Event()

    class _HangingGraph:
        async def astream(self, state):
            started.set()
            await asyncio.Event().wait()  # never set
            yield {}  # pragma: no cover

    monkeypatch.setattr(runner_module, "create_negotiation_graph", lambda: _HangingGraph())

    neg_id, initial = await prepare_negotiation(db, _REQUEST)
    task = asyncio.create_task(run_negotiation(initial, neg_id, db))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    record = await db.get_negotiation(neg_id)
    assert record["status"] == "failed"
    # The request terms survive: the upsert must not NULL them.
    assert record["max_budget"] == 500.0
    assert record["min_price"] == 100.0


async def test_wall_clock_timeout_cancels_the_in_flight_llm_call(db, monkeypatch):
    """The LLM node is async, so the timeout cancels the request itself.

    As a sync node it ran in a worker thread that kept the provider call alive
    after the negotiation had already been recorded as timed out.
    """
    import agent_bargain.agents.base as base_module
    from agent_bargain.config import settings

    cancelled = asyncio.Event()

    class _SlowStructured:
        async def ainvoke(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class _SlowLLM:
        def with_structured_output(self, schema):
            return _SlowStructured()

    monkeypatch.setattr(base_module, "get_negotiation_llm", lambda *a, **k: _SlowLLM())
    monkeypatch.setattr(settings, "negotiation_timeout_seconds", 0.2)

    neg_id, initial = await prepare_negotiation(db, _REQUEST)
    summary = await asyncio.wait_for(run_negotiation(initial, neg_id, db), timeout=5)

    assert summary["status"] == "timeout"
    assert cancelled.is_set(), "the LLM call outlived the negotiation"
