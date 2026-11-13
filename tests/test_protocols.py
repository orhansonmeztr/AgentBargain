"""Tests for the MCP tool API (schemas + directly callable handlers).

The MCP transport (``protocols/mcp_server.py``) is covered separately in
``test_mcp_server.py``; the A2A server is covered in ``test_a2a_server.py``.
These tests pin the *Python API* surface the transport and an in-process
tool-calling agent both bind to.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_bargain.engine.discovery import DiscoveryEngine
from agent_bargain.engine.reputation import ReputationEngine
from agent_bargain.models.negotiation import Agreement
from agent_bargain.models.reputation import FeedbackScore
from agent_bargain.models.service import ServiceListing
from agent_bargain.protocols import (
    handle_query_reputation,
    handle_search_services,
)
from agent_bargain.storage.database import DatabaseManager
from agent_bargain.storage.vector_store import VectorStoreManager

# -- MCP handlers -----------------------------------------------------------


async def test_handle_query_reputation_reads_the_ledger(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "mcp.db")
    await db.connect()
    try:
        # Foreign keys are enforced, so the referenced rows must exist first.
        await db.ensure_agent("buyer_1")
        await db.ensure_agent("seller_1")
        await db.upsert_service(
            ServiceListing(
                service_id="svc_1",
                seller_id="seller_1",
                title="Test Service",
                base_price=100.0,
                sla_terms="Standard",
            )
        )
        await db.save_agreement(
            Agreement(
                agreement_id="agr_1",
                buyer_id="buyer_1",
                seller_id="seller_1",
                service_id="svc_1",
                final_price=100.0,
                final_sla="Standard",
            )
        )
        await db.save_feedback(
            FeedbackScore(
                feedback_id="fb_1",
                agreement_id="agr_1",
                evaluator_id="system_evaluator",
                target_id="seller_1",
                score=5.0,
                comments="excellent",
            )
        )

        result = await handle_query_reputation("seller_1", db, ReputationEngine())
    finally:
        await db.close()

    assert result == {"agent_id": "seller_1", "score": 5.0, "total_transactions": 1}


async def test_handle_query_reputation_unknown_agent_is_neutral(tmp_path):
    """An agent with no history reports the neutral score, not 0.0."""
    db = DatabaseManager(db_path=tmp_path / "mcp2.db")
    await db.connect()
    try:
        result = await handle_query_reputation("nobody", db, ReputationEngine())
    finally:
        await db.close()

    assert result["total_transactions"] == 0
    assert result["score"] == ReputationEngine().generate_record("nobody", []).current_score


def test_handle_search_services_delegates_to_discovery():
    store = MagicMock(spec=VectorStoreManager)
    store.search_services.return_value = {"ids": [["svc_1"]]}
    store.get_metadatas.return_value = [{"title": "Web Dev"}]

    engine = DiscoveryEngine(store)
    engine.index_services(["svc_1"], ["expert web development"], [{"title": "Web Dev"}])

    results = handle_search_services("web development", engine, top_k=3)

    assert [r["id"] for r in results] == ["svc_1"]
    assert results[0]["metadata"] == {"title": "Web Dev"}
    assert "rrf_score" in results[0]
