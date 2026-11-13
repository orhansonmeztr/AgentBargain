"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent_bargain.api.auth import (
    require_auth_when_exposed,
    verify_api_key,
    warn_if_unauthenticated,
)
from agent_bargain.api.routes import analytics, marketplace, negotiations, websocket
from agent_bargain.api.routes.websocket import ConnectionManager
from agent_bargain.config import configure_langsmith, configure_logging, settings
from agent_bargain.engine.discovery import DiscoveryEngine, corpus_from_rows
from agent_bargain.protocols.a2a_server import add_a2a_to_app
from agent_bargain.storage.database import DatabaseManager
from agent_bargain.storage.vector_store import create_vector_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    # Startup
    # Logging first: warn_if_unauthenticated and every other startup line are
    # discarded if the root logger is still at Python's WARNING default.
    configure_logging()
    configure_langsmith()
    # Before anything is opened or served: an exposed, unauthenticated API
    # is refused rather than merely logged.
    require_auth_when_exposed(settings.server_host)
    warn_if_unauthenticated(settings.server_host)

    db = DatabaseManager()
    await db.connect()

    # Recover negotiations orphaned by a previous crash or restart. Must run
    # before serving: an ``active`` row at this point cannot belong to a live
    # task, and leaving it would make it indistinguishable from one that is.
    orphans = await db.fail_orphaned_negotiations()
    if orphans:
        logger.warning(
            "Marked %d orphaned negotiation(s) as 'failed' — a previous process "
            "died before finishing them.",
            orphans,
        )

    vector_store = create_vector_store()
    discovery = DiscoveryEngine(vector_store)

    # Seed the hybrid discovery index with any already-persisted services.
    services = await db.list_all_services()
    if services:
        discovery.index_services(*corpus_from_rows(services))

    app.state.db = db
    app.state.discovery = discovery
    # Per-app rather than a module singleton, so two create_app() instances
    # (tests, embedding) cannot broadcast into each other's clients.
    app.state.ws_manager = ConnectionManager()

    yield

    # Shutdown
    await db.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AgentBargain API",
        version="0.1.0",
        description=(
            "A reputation-based marketplace for autonomous Multi-Agent Systems. "
            "Agents discover each other, negotiate terms using LangGraph state machines, "
            "and build reputation through time-decaying feedback scores."
        ),
        contact={"name": "Orhan Sonmez", "url": "https://github.com/orhansonmeztr"},
        license_info={"name": "MIT"},
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        # The key travels in X-API-Key, so it must be an allowed request header.
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check() -> dict:
        # Unauthenticated on purpose: container health checks and load balancers
        # probe it, and it reveals nothing but liveness.
        return {"status": "ok"}

    # Every /api/v1 HTTP route requires a key when API_KEYS is configured.
    # One deliberate exception, explained in api/auth.py: the WebSocket route,
    # which authenticates during its own handshake — a router-level dependency
    # would be applied to it too, and a dependency taking ``Request`` cannot
    # resolve for a WebSocket. The A2A JSON-RPC endpoint (mounted below) is
    # guarded by its own middleware; only its Agent Card is public.
    protected = [Depends(verify_api_key)]
    app.include_router(
        negotiations.router,
        prefix="/api/v1/negotiations",
        tags=["Negotiations"],
        dependencies=protected,
    )
    app.include_router(
        marketplace.router,
        prefix="/api/v1/marketplace",
        tags=["Marketplace"],
        dependencies=protected,
    )
    app.include_router(
        analytics.router,
        prefix="/api/v1/analytics",
        tags=["Analytics"],
        dependencies=protected,
    )
    app.include_router(websocket.router, prefix="/api/v1/ws", tags=["WebSocket"])

    # A2A (Agent-to-Agent) Agent Card + JSON-RPC task endpoints.
    add_a2a_to_app(app)

    # Mount static dashboard last so it does not override API routes.
    # Resolved relative to this file, not PROJECT_ROOT: an installed wheel has no
    # <root>/src/agent_bargain/dashboard, and mkdir would silently create an
    # empty directory to serve instead of failing loudly.
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard"
    if not dashboard_path.is_dir():
        raise RuntimeError(
            f"Dashboard assets not found at {dashboard_path}. The package appears "
            "to be installed without its static files."
        )
    app.mount("/", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")

    return app
