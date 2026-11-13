"""Protocol adapters for A2A and MCP integration.

* **A2A** — :mod:`~agent_bargain.protocols.a2a_server` exposes the negotiation
  engine as a real Agent-to-Agent server using the official ``a2a-sdk``. It
  serves a spec-compliant Agent Card and JSON-RPC ``message/send``, mapping each
  task to a negotiation.
* **MCP** — :mod:`~agent_bargain.protocols.mcp_tools` provides the tool descriptions
  and directly callable handlers; :mod:`~agent_bargain.protocols.mcp_server`
  wraps them in a :class:`MCPServer` and a stdio entry point
  (``agent-bargain-mcp``).
"""

from agent_bargain.protocols.a2a_server import (
    add_a2a_to_app,
    build_agent_card,
)
from agent_bargain.protocols.mcp_server import create_mcp_server
from agent_bargain.protocols.mcp_tools import (
    handle_query_reputation,
    handle_search_services,
)

__all__ = [
    "add_a2a_to_app",
    "build_agent_card",
    "create_mcp_server",
    "handle_query_reputation",
    "handle_search_services",
]
