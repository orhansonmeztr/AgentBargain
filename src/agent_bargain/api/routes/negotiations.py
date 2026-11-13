"""Negotiation routing endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request

from agent_bargain.agents.guardrails import public_text
from agent_bargain.api.auth import can_read_reasoning
from agent_bargain.api.routes.websocket import ConnectionManager
from agent_bargain.engine.errors import NegotiationSetupError
from agent_bargain.engine.negotiation_runner import prepare_negotiation, run_negotiation
from agent_bargain.models.negotiation import NegotiationRequest
from agent_bargain.storage.database import DatabaseManager

router = APIRouter()
logger = logging.getLogger(__name__)

#: Each party's walk-away point. ``min_price`` is the seller's floor (hidden from
#: listing responses for the same reason) and ``max_budget`` the buyer's ceiling;
#: every key can read every negotiation, so neither may appear in list views or
#: one side learns the other's limit for its next negotiation on the listing.
_PRIVATE_NEGOTIATION_FIELDS = ("min_price", "max_budget")


def _public_negotiation(row: dict[str, Any]) -> dict[str, Any]:
    """Strip both parties' private limits from a negotiation row."""
    return {k: v for k, v in row.items() if k not in _PRIVATE_NEGOTIATION_FIELDS}


async def _run_negotiation(
    state: dict[str, Any],
    negotiation_id: str,
    db: DatabaseManager,
    ws_manager: ConnectionManager,
) -> None:
    """Run a negotiation in the background, streaming its events to the dashboard.

    Thin adapter over :func:`~agent_bargain.engine.negotiation_runner.run_negotiation`.
    It exists so the background task is handed the app-scoped ``db`` and
    ``ws_manager`` explicitly, which ``test_api.py`` asserts.
    """
    await run_negotiation(state, negotiation_id, db, broadcast=ws_manager.broadcast_event)


@router.post("/start")
async def start_negotiation(
    body: NegotiationRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict[str, Any]:
    """Start a new negotiation between buyer and seller.

    The service must already be registered by its seller; the seller's floor
    price is read from that listing rather than taken from the request body.
    """
    db: DatabaseManager = request.app.state.db

    try:
        negotiation_id, initial = await prepare_negotiation(db, body)
    except NegotiationSetupError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    background_tasks.add_task(
        _run_negotiation,
        initial,
        negotiation_id,
        db,
        request.app.state.ws_manager,
    )

    return {"status": "started", "negotiation_id": negotiation_id}


@router.get("/history")
async def list_negotiations(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List recent negotiations (most recent first)."""
    db: DatabaseManager = request.app.state.db
    records = await db.list_negotiations(limit=limit, offset=offset)
    return {
        "negotiations": [_public_negotiation(r) for r in records],
        "limit": limit,
        "offset": offset,
    }


@router.get("/{negotiation_id}/status")
async def get_negotiation_status(
    negotiation_id: str,
    request: Request,
) -> dict[str, Any]:
    """Get the current status of a negotiation from the database.

    Returns 404 if the negotiation ID is not found. Clients should distinguish
    between "not found" and "still running" — only IDs returned by POST /start
    can ever be active; unknown IDs are always 404.
    """
    db: DatabaseManager = request.app.state.db
    record = await db.get_negotiation(negotiation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Negotiation '{negotiation_id}' not found")
    return {
        "negotiation_id": record["negotiation_id"],
        "status": record["status"],
        "rounds": record["rounds"],
        "final_price": record["final_price"],
        "final_sla": record["final_sla"],
        "buyer_id": record["buyer_id"],
        "seller_id": record["seller_id"],
        "service_id": record["service_id"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


@router.get("/{negotiation_id}/messages")
async def get_negotiation_messages(
    negotiation_id: str,
    request: Request,
) -> dict[str, Any]:
    """Get all messages exchanged during a negotiation (for the dashboard chat view)."""
    db: DatabaseManager = request.app.state.db
    # Same contract as /status: an unknown id is 404, not an empty transcript
    # that looks like a negotiation which has not started talking yet.
    if await db.get_negotiation(negotiation_id) is None:
        raise HTTPException(status_code=404, detail=f"Negotiation '{negotiation_id}' not found")
    messages = await db.get_messages_for_negotiation(negotiation_id)
    if not can_read_reasoning(request.headers):
        messages = [_public_message(m) for m in messages]
    return {"negotiation_id": negotiation_id, "messages": messages}


def _public_message(row: dict[str, Any]) -> dict[str, Any]:
    """A transcript row without the negotiator's reasoning.

    LLMs state their own limit in their reasoning ("my minimum acceptable
    price is $80"), and any regular key can act as either party, so only
    admin keys see it — like the floor and budget themselves.
    """
    public = dict(row)
    public["content"] = public_text(str(row.get("content") or ""))
    raw = row.get("raw_data")
    if raw:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            public["raw_data"] = None
        else:
            if isinstance(data, dict):
                data.pop("reasoning", None)
                public["raw_data"] = json.dumps(data)
            else:
                public["raw_data"] = None
    return public
