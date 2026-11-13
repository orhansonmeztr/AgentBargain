"""MCP tool descriptions and their executable handlers."""

from __future__ import annotations

from typing import Any

from agent_bargain.engine.discovery import DiscoveryEngine
from agent_bargain.engine.reputation import ReputationEngine
from agent_bargain.storage.database import DatabaseManager

# -- Tool descriptions ------------------------------------------------------
#
# Shared with the MCP server in :mod:`agent_bargain.protocols.mcp_server`, which
# derives each tool's input schema from its wrapper's signature.

QUERY_REPUTATION_DESCRIPTION = "Query the reputation score of a specific agent."
SEARCH_SERVICES_DESCRIPTION = "Search the marketplace for available services."

#: ``search_services`` result limits, shared with the MCP server and matching
#: the HTTP search route's bound.
DEFAULT_TOP_K = 5
MAX_TOP_K = 100


# -- Tool handlers (callable implementations) -----------------------------


async def handle_query_reputation(
    agent_id: str,
    db: DatabaseManager,
    reputation_engine: ReputationEngine,
) -> dict[str, Any]:
    """Execute the query_reputation tool and return the result."""
    feedbacks = await db.get_feedbacks_for_agent(agent_id)
    record = reputation_engine.generate_record(agent_id, feedbacks)
    return {
        "agent_id": record.agent_id,
        "score": record.current_score,
        "total_transactions": record.total_transactions,
    }


def handle_search_services(
    query: str,
    discovery_engine: DiscoveryEngine,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Execute the search_services tool and return ranked results.

    ``top_k`` is clamped to ``[1, MAX_TOP_K]``: discovery fetches ``2 * top_k``
    candidates from each ranker, so an unbounded value is a cheap way to make
    every call scan and serialise the whole corpus.
    """
    return discovery_engine.search(query, top_k=max(1, min(MAX_TOP_K, top_k)))
