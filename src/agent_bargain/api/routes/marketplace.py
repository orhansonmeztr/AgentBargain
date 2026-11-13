"""Marketplace discovery endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from agent_bargain.engine.discovery import service_document, service_metadata
from agent_bargain.models.service import ServiceCategory, ServiceListing
from agent_bargain.storage.database import DatabaseManager

router = APIRouter()

#: ``floor_price`` is the seller's walk-away point. It must never appear in a
#: listing response, or any buyer can read the floor before negotiating and the
#: whole exercise becomes theatre.
_PRIVATE_SERVICE_FIELDS = ("floor_price",)


def _public_service(row: dict[str, Any]) -> dict[str, Any]:
    """Strip seller-private fields from a service row before returning it."""
    return {k: v for k, v in row.items() if k not in _PRIVATE_SERVICE_FIELDS}


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)


class RegisterServiceRequest(BaseModel):
    """A seller publishes a service, including its own floor price.

    The floor is set *here*, by the seller, and never by a negotiating
    counterparty. ``POST /negotiations/start`` reads it back from this listing.
    """

    service_id: str = Field(min_length=1)
    seller_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = "No description provided."
    category_id: str = "uncategorized"
    category_name: str = "Uncategorized"
    base_price: float = Field(ge=0.0, description="Advertised asking price")
    floor_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Seller's minimum acceptable price; hidden from buyers",
    )
    sla_terms: str = "Standard"

    @model_validator(mode="after")
    def _floor_not_above_asking_price(self) -> RegisterServiceRequest:
        # A floor above the asking price makes every default negotiation open
        # below the floor, which POST /negotiations/start rejects with 422 —
        # the listing could never be negotiated without an explicit
        # initial_price. Reject the contradiction where it is introduced.
        if self.floor_price > self.base_price:
            raise ValueError("floor_price must not exceed base_price")
        return self


@router.post("/services", status_code=201)
async def register_service(body: RegisterServiceRequest, request: Request) -> dict[str, Any]:
    """Register (or update) a service listing.

    Only the seller who registered a ``service_id`` can update it; a different
    ``seller_id`` gets 409.

    Required before a negotiation can reference the service: the seller's floor
    has to exist somewhere the buyer cannot write to.
    """
    db: DatabaseManager = request.app.state.db

    # Updating a listing is the owner's prerogative. The upsert used to accept
    # any seller_id, so re-registering someone else's service_id silently
    # transferred it — new owner, new floor — under every buyer's feet.
    conflict = HTTPException(
        status_code=409,
        detail=(
            f"Service '{body.service_id}' is already registered by another "
            "seller; choose a different service_id."
        ),
    )
    # Fast path; the authoritative check is inside upsert_service below.
    existing = await db.get_service(body.service_id)
    if existing is not None and existing.seller_id != body.seller_id:
        raise conflict

    # Foreign keys are enforced, so the seller must exist before the listing.
    await db.ensure_agent(body.seller_id)

    service = ServiceListing(
        service_id=body.service_id,
        seller_id=body.seller_id,
        title=body.title,
        description=body.description,
        category=ServiceCategory(id=body.category_id, name=body.category_name),
        base_price=body.base_price,
        floor_price=body.floor_price,
        sla_terms=body.sla_terms,
    )
    # The ownership check is atomic with the write, so a concurrent
    # registration by another seller between the read above and here still
    # cannot take the listing over.
    if not await db.upsert_service(service):
        raise conflict

    # Index for hybrid discovery. The request body has the same shape as a
    # service row, so it goes through the same helpers the startup seed uses;
    # neither puts the floor price into the document or the metadata.
    row = body.model_dump()
    await run_in_threadpool(
        request.app.state.discovery.add_service,
        service.service_id,
        service_document(row),
        service_metadata(row),
    )

    return {"status": "registered", "service": _public_service(service.model_dump())}


@router.get("/services/{service_id}")
async def get_service(service_id: str, request: Request) -> dict[str, Any]:
    """Fetch one service listing, without the seller's floor price."""
    db: DatabaseManager = request.app.state.db
    service = await db.get_service(service_id)
    if service is None:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not registered")
    return _public_service(service.model_dump())


@router.post("/search")
async def search_services(body: SearchRequest, request: Request) -> dict[str, Any]:
    """Search for services using the hybrid discovery engine (BM25 + Vector).

    Discovery performs blocking ChromaDB I/O and CPU-bound BM25 scoring, so it
    runs in a worker thread. Called directly it would stall the event loop —
    and with it every WebSocket client and concurrent request.
    """
    discovery = request.app.state.discovery
    results: list[dict[str, Any]] = await run_in_threadpool(
        discovery.search, body.query, body.top_k
    )
    return {"results": results, "query": body.query, "count": len(results)}


@router.get("/services")
async def list_services(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List all registered services from the database."""
    db: DatabaseManager = request.app.state.db
    services = await db.list_services(limit=limit, offset=offset)
    return {
        "services": [_public_service(s) for s in services],
        "count": len(services),
    }


@router.get("/agents")
async def list_agents(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List all registered agents ordered by reputation score (leaderboard)."""
    db: DatabaseManager = request.app.state.db
    agents = await db.list_agents(limit=limit, offset=offset)
    return {"agents": agents, "count": len(agents)}
