"""Standalone MCP (Model Context Protocol) server over stdio.

This module turns the Python-only tool handlers in
:mod:`agent_bargain.protocols.mcp_tools` into a real, connectable MCP server: a
process an MCP client (Claude Desktop, an agent harness, ...) can launch over
the stdio transport and call ``query_reputation`` / ``search_services`` on.

The tool *implementations* live in ``mcp_tools.py``; this module is only the
transport and the wiring. It builds the same dependencies the FastAPI app
does — a :class:`~agent_bargain.storage.database.DatabaseManager` and a
:class:`~agent_bargain.engine.discovery.DiscoveryEngine` — inside an MCP
lifespan, and hands them to the handlers on each tool call.

.. note::
   This is a **read-only** surface, and it must be safe to run concurrently
   with a live API server sharing the same files. The SQLite database is opened
   with ``mode=ro`` (no DDL, no journal-mode switch, writes raise), and the
   persisted services are loaded into the in-memory BM25 corpus only — the
   Chroma directory belongs to the API server and is never upserted from here.
   Vector search therefore reflects what the API server last indexed, and the
   database must already exist (start the API server once first).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from agent_bargain import __version__
from agent_bargain.engine.discovery import DiscoveryEngine, corpus_from_rows
from agent_bargain.engine.reputation import ReputationEngine
from agent_bargain.protocols.mcp_tools import (
    DEFAULT_TOP_K,
    QUERY_REPUTATION_DESCRIPTION,
    SEARCH_SERVICES_DESCRIPTION,
    handle_query_reputation,
    handle_search_services,
)
from agent_bargain.storage.database import DatabaseManager
from agent_bargain.storage.vector_store import create_vector_store

logger = logging.getLogger(__name__)

#: Server name advertised in the MCP handshake.
SERVER_NAME = "agent-bargain"

SERVER_INSTRUCTIONS = (
    "AgentBargain marketplace tools. Use query_reputation to look up an agent's "
    "time-decaying reputation score and transaction count, and search_services "
    "to discover available service listings via hybrid (BM25 + vector) search."
)


@dataclass
class MCPServerState:
    """Dependencies carried through the lifespan and injected into tool calls."""

    db: DatabaseManager
    reputation: ReputationEngine
    discovery: DiscoveryEngine


def _seed_discovery(discovery: DiscoveryEngine, services: list[dict[str, Any]]) -> None:
    """Load already-persisted services so ``search_services`` has data at startup.

    Unlike the FastAPI lifespan this uses ``load_corpus``, which leaves the
    shared, persisted vector store untouched.
    """
    if services:
        discovery.load_corpus(*corpus_from_rows(services))


@asynccontextmanager
async def mcp_lifespan(server: MCPServer) -> AsyncIterator[MCPServerState]:
    """Build and tear down the read-only dependencies each tool needs.

    ``server`` is unused by this lifespan (the tools reach their dependencies
    through the yielded context), but MCP requires this signature.
    """
    db = DatabaseManager()
    await db.connect(read_only=True)

    discovery = DiscoveryEngine(create_vector_store())
    _seed_discovery(discovery, await db.list_all_services())

    state = MCPServerState(db=db, reputation=ReputationEngine(), discovery=discovery)
    logger.info("AgentBargain MCP server started (tools=query_reputation, search_services).")
    try:
        yield state
    finally:
        await db.close()


async def _query_reputation(agent_id: str, ctx: Context) -> dict[str, Any]:
    """MCP tool wrapper around ``handle_query_reputation``."""
    state: MCPServerState = ctx.request_context.lifespan_context
    return await handle_query_reputation(agent_id, state.db, state.reputation)


def _search_services(
    query: str,
    ctx: Context,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """MCP tool wrapper around ``handle_search_services``."""
    state: MCPServerState = ctx.request_context.lifespan_context
    return handle_search_services(query, state.discovery, top_k=top_k)


def create_mcp_server() -> MCPServer:
    """Create and fully wire the AgentBargain MCP server.

    Kept as a factory rather than a module-level singleton so that constructing
    the server (which configures MCP logging) never happens at import time —
    tests can import this module without side effects, matching the FastAPI
    app's "configure logging at startup, not import" rule.
    """
    server = MCPServer(
        name=SERVER_NAME,
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
        lifespan=mcp_lifespan,
    )
    server.add_tool(
        _query_reputation,
        name="query_reputation",
        description=QUERY_REPUTATION_DESCRIPTION,
    )
    server.add_tool(
        _search_services,
        name="search_services",
        description=SEARCH_SERVICES_DESCRIPTION,
    )
    return server


def main() -> None:
    """Entry point for the ``agent-bargain-mcp`` console script (stdio)."""
    create_mcp_server().run()
