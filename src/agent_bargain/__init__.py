"""
AgentBargain - A reputation-based marketplace for autonomous Multi-Agent Systems.

A simulated marketplace where autonomous AI agents discover each other,
negotiate terms, and execute digital service transactions.
"""

__version__ = "0.1.0"


def main() -> None:
    """Entry point for the agent-bargain CLI."""
    import uvicorn

    from agent_bargain.config import settings

    uvicorn.run(
        "agent_bargain.api.app:create_app",
        factory=True,
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )
