"""Tests for the standalone MCP (Model Context Protocol) stdio server.

These pin the transport layer (``protocols/mcp_server.py``): that the two
handlers from ``protocols/mcp_tools.py`` are registered with the right names and
input schemas, and that an end-to-end ``call_tool`` reaches them with the
dependencies the lifespan provides.

The lifespan itself is deliberately *not* run here. It would open the real
SQLite/Chroma paths from the developer's ``.env``; instead each test builds an
``MCPServerState`` by hand with isolated storage, matching ``test_protocols.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import Context

from agent_bargain import __version__
from agent_bargain.engine.discovery import DiscoveryEngine
from agent_bargain.engine.reputation import ReputationEngine
from agent_bargain.models.negotiation import Agreement
from agent_bargain.models.reputation import FeedbackScore
from agent_bargain.models.service import ServiceListing
from agent_bargain.protocols.mcp_server import (
    SERVER_NAME,
    MCPServerState,
    create_mcp_server,
)
from agent_bargain.protocols.mcp_tools import (
    QUERY_REPUTATION_DESCRIPTION,
    SEARCH_SERVICES_DESCRIPTION,
)
from agent_bargain.storage.database import DatabaseManager


def _call_context(server, state: MCPServerState) -> Context:
    """A Context carrying ``state`` as its lifespan context, as a real request would."""
    return Context(
        request_context=ServerRequestContext(
            session=MagicMock(),
            lifespan_context=state,
            protocol_version="2025-06-18",
            method="tools/call",
        ),
        mcp_server=server,
    )


def _structured(result):
    """Return the tool's payload from ``structured_content``, unwrapping MCP's ``{"result": ...}``.

    MCP's structured-output convention keeps ``dict[str, ...]`` returns as-is but
    wraps list returns in ``{"result": [...]}`` (see ``_create_output_model``),
    so a uniform assertion has to undo the wrap.
    """
    assert not result.is_error, result.content
    content = result.structured_content
    if isinstance(content, dict) and tuple(content) == ("result",):
        return content["result"]
    return content


# -- Registration -----------------------------------------------------------


def test_server_identity_and_version():
    server = create_mcp_server()
    assert server.name == SERVER_NAME
    assert server.version == __version__


async def test_tools_are_registered_with_matching_descriptions():
    server = create_mcp_server()
    tools = {tool.name: tool for tool in await server.list_tools()}

    assert set(tools) == {"query_reputation", "search_services"}
    assert tools["query_reputation"].description == QUERY_REPUTATION_DESCRIPTION
    assert tools["search_services"].description == SEARCH_SERVICES_DESCRIPTION


async def test_tool_input_schemas_exclude_the_context_parameter():
    server = create_mcp_server()
    tools = {tool.name: tool for tool in await server.list_tools()}

    # The injected MCP Context must not leak into the advertised input schema.
    query_props = tools["query_reputation"].input_schema["properties"]
    assert query_props == {"agent_id": {"title": "Agent Id", "type": "string"}}
    assert tools["query_reputation"].input_schema["required"] == ["agent_id"]

    search_props = tools["search_services"].input_schema["properties"]
    assert set(search_props) == {"query", "top_k"}
    assert search_props["top_k"]["default"] == 5
    assert tools["search_services"].input_schema["required"] == ["query"]


# -- End-to-end call_tool ---------------------------------------------------


async def test_call_query_reputation_reads_the_ledger(tmp_path):
    server = create_mcp_server()
    db = DatabaseManager(db_path=tmp_path / "mcp_server.db")
    await db.connect()
    try:
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

        state = MCPServerState(db=db, reputation=ReputationEngine(), discovery=MagicMock())
        result = await server.call_tool(
            "query_reputation", {"agent_id": "seller_1"}, _call_context(server, state)
        )
    finally:
        await db.close()

    assert _structured(result) == {
        "agent_id": "seller_1",
        "score": 5.0,
        "total_transactions": 1,
    }


async def test_call_query_reputation_unknown_agent_is_neutral(tmp_path):
    server = create_mcp_server()
    db = DatabaseManager(db_path=tmp_path / "mcp_server_unknown.db")
    await db.connect()
    try:
        state = MCPServerState(db=db, reputation=ReputationEngine(), discovery=MagicMock())
        result = await server.call_tool(
            "query_reputation", {"agent_id": "nobody"}, _call_context(server, state)
        )
    finally:
        await db.close()

    assert _structured(result) == {"agent_id": "nobody", "score": 3.0, "total_transactions": 0}


async def test_call_search_services_delegates_with_top_k():
    server = create_mcp_server()
    discovery = MagicMock(spec=DiscoveryEngine)
    discovery.search.return_value = [
        {"id": "svc_1", "metadata": {"title": "Web Dev"}, "rrf_score": 0.5}
    ]

    state = MCPServerState(db=MagicMock(), reputation=ReputationEngine(), discovery=discovery)
    result = await server.call_tool(
        "search_services",
        {"query": "web development", "top_k": 3},
        _call_context(server, state),
    )

    assert _structured(result) == [
        {"id": "svc_1", "metadata": {"title": "Web Dev"}, "rrf_score": 0.5}
    ]
    discovery.search.assert_called_once_with("web development", top_k=3)


def test_seeding_never_writes_the_shared_vector_store():
    """The Chroma directory belongs to the API server; MCP only builds BM25."""
    from agent_bargain.protocols.mcp_server import _seed_discovery

    store = MagicMock()
    store.get_metadatas.side_effect = lambda ids: [None for _ in ids]  # not indexed there
    store.search_services.return_value = {"ids": [[]]}
    discovery = DiscoveryEngine(store)

    _seed_discovery(
        discovery,
        [{"service_id": "svc_1", "title": "GPU cluster", "seller_id": "s1"}],
    )

    store.add_services.assert_not_called()
    # BM25 still finds it, and metadata falls back to the in-memory mirror.
    results = discovery.search("gpu", top_k=1)
    assert [r["id"] for r in results] == ["svc_1"]
    assert results[0]["metadata"]["title"] == "GPU cluster"


async def test_search_schema_advertises_top_k_and_handler_clamps_it():
    from agent_bargain.protocols.mcp_tools import MAX_TOP_K, handle_search_services

    tools = {tool.name: tool for tool in await create_mcp_server().list_tools()}
    assert "top_k" in tools["search_services"].input_schema["properties"]

    discovery = MagicMock()
    handle_search_services("q", discovery, top_k=10_000)
    discovery.search.assert_called_once_with("q", top_k=MAX_TOP_K)
