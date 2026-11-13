"""WebSocket event streaming."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette import status

from agent_bargain.api.auth import can_read_reasoning, verify_websocket_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting.

    One instance lives on ``app.state.ws_manager``, created in the FastAPI
    lifespan. It used to be a module-level singleton, which meant every
    ``create_app()`` shared one connection list — a client connected to one app
    instance received events from another.

    .. note::
       This is per *process*. Running several Uvicorn workers gives each its own
       manager, so a client only sees negotiations started in the worker it
       happens to be connected to. Fan-out across workers needs a shared bus
       (e.g. Redis pub/sub) and is not implemented. The default single-process
       entry point (``agent-bargain``) is unaffected.
    """

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        #: Connections authenticated with an admin key (or with auth off).
        #: Only these receive negotiators' reasoning; see ``broadcast_event``.
        self.reasoning_readers: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, *, can_read_reasoning: bool = False) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        if can_read_reasoning:
            self.reasoning_readers.add(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.reasoning_readers.discard(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self.active_connections))

    async def broadcast(self, message: str, *, public_message: str | None = None) -> None:
        """Send a message to all connected clients, removing dead connections.

        With ``public_message``, only reasoning readers get ``message``; every
        other client gets ``public_message`` instead.
        """
        # Snapshot first: broadcast awaits on each send, and another coroutine
        # (a concurrent negotiation, or a client disconnecting) may mutate the
        # list in the meantime — iterating it live would raise.
        dead: list[WebSocket] = []
        for connection in list(self.active_connections):
            text = (
                message
                if public_message is None or connection in self.reasoning_readers
                else public_message
            )
            try:
                await connection.send_text(text)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)

    async def broadcast_event(self, event: dict[str, Any]) -> None:
        """Serialise an event dict and broadcast it.

        Matches ``run_negotiation``'s ``broadcast`` callback, so both the HTTP
        route and the A2A executor pass this bound method instead of each
        wrapping ``broadcast`` in its own closure.
        """
        full, public = _split_event(event)
        await self.broadcast(
            json.dumps(full),
            public_message=None if public is None else json.dumps(public),
        )


def _split_event(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(full, public)`` forms of an event; ``public`` is ``None`` if they match.

    Turn events carry the negotiator's free-text reasoning, which routinely
    states its own limit ("my budget is $300"). Regular keys can act as any
    party, so they get the turn without it — admin keys, as for the
    game-theory analytics, get everything.
    """
    full = {k: v for k, v in event.items() if k != "public_content"}
    if "reasoning" not in event and "public_content" not in event:
        return full, None
    public = dict(full)
    public["reasoning"] = ""
    if "public_content" in event:
        public["content"] = event["public_content"]
    return full, public


@router.websocket("/events")
async def websocket_endpoint(
    websocket: WebSocket,
    api_key: str | None = Query(default=None),
) -> None:
    """WebSocket endpoint for real-time monitoring of agent events.

    Authenticated during the handshake rather than by a router-level dependency:
    once ``accept()`` has been called the connection can no longer be refused
    with an HTTP 401. Browsers cannot set headers on ``new WebSocket()``, so the
    key may also arrive as ``?api_key=`` — which places it in the URL and any
    access log, and should be treated as the weaker of the two transports.
    """
    if not verify_websocket_api_key(websocket, api_key):
        # 1008 = policy violation. Closing before accept() rejects the handshake.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning(
            "Rejected WebSocket handshake with a missing or invalid API key from %s",
            websocket.client,
        )
        return

    manager: ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(
        websocket, can_read_reasoning=can_read_reasoning(websocket.headers, api_key)
    )
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
