"""A2A (Agent-to-Agent) server backed by the ``a2a-sdk``.

Exposes the AgentBargain negotiation engine as a real A2A agent: an A2A client
discovers the agent via its Agent Card, then sends a ``message/send`` whose text
is a JSON-encoded :class:`~agent_bargain.models.negotiation.NegotiationRequest`.
The server runs the same negotiation the HTTP ``POST /negotiations/start`` route
does and returns the outcome as a task artifact.

Authentication: the JSON-RPC endpoint requires the same API key as the rest
of ``/api/v1`` whenever keys are configured (:class:`A2AAuthMiddleware`). It
starts negotiations and writes to the reputation ledger, so leaving it open
made ``API_KEYS`` a lock with the back door propped open. The Agent Card stays
public — discovery is its purpose — and advertises the ``X-API-Key`` scheme in
``security_schemes`` so A2A clients know what to send.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from a2a.helpers.proto_helpers import (
    get_data_parts,
    get_message_text,
    new_data_artifact_update_event,
    new_task,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    APIKeySecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    TaskState,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_1_0
from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from agent_bargain.api.auth import API_KEY_HEADER, auth_enabled, headers_authenticate
from agent_bargain.engine.errors import NegotiationSetupError
from agent_bargain.engine.negotiation_runner import prepare_negotiation, run_negotiation
from agent_bargain.models.negotiation import NegotiationRequest

logger = logging.getLogger(__name__)

#: URL prefix under which the A2A surface is mounted inside the FastAPI app.
A2A_PREFIX = "/api/v1/a2a"
A2A_CARD_URL = f"{A2A_PREFIX}{AGENT_CARD_WELL_KNOWN_PATH}"
A2A_RPC_URL = f"{A2A_PREFIX}/jsonrpc"

#: Name/version advertised in the A2A Agent Card.
AGENT_NAME = "AgentBargain Negotiation Agent"
AGENT_VERSION = "1.0.0"

#: Name of the security scheme advertised on the Agent Card.
API_KEY_SCHEME = "apiKey"


def build_agent_card(public_base_url: str = "") -> AgentCard:
    """Build the spec-compliant A2A Agent Card describing the negotiation skill.

    ``public_base_url``, when supplied, is prepended to the JSON-RPC endpoint so
    the card's interface URL is absolute; otherwise a relative path is used
    (same-origin clients resolve it against the card URL).
    """
    rpc_url = f"{public_base_url}{A2A_RPC_URL}" if public_base_url else A2A_RPC_URL

    # Advertise the key only when one is actually required, so a card never
    # promises a scheme the server does not enforce (or omits one it does).
    security: dict[str, Any] = {}
    if auth_enabled():
        security = {
            "security_schemes": {
                API_KEY_SCHEME: SecurityScheme(
                    api_key_security_scheme=APIKeySecurityScheme(
                        description="Same API key as the rest of /api/v1.",
                        location="header",
                        name=API_KEY_HEADER,
                    )
                )
            },
            "security_requirements": [
                SecurityRequirement(schemes={API_KEY_SCHEME: StringList(list=[])})
            ],
        }

    return AgentCard(
        **security,
        name=AGENT_NAME,
        description=(
            "Negotiates service deals in the AgentBargain reputation-based "
            "marketplace. Send a JSON-encoded negotiation request to open a "
            "buyer/seller bargain."
        ),
        version=AGENT_VERSION,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        supported_interfaces=[
            AgentInterface(
                url=rpc_url,
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        skills=[
            AgentSkill(
                id="negotiate",
                name="Negotiate a service deal",
                description=(
                    "Open a negotiation over an already-registered service. The "
                    "input is a JSON object with buyer_id, seller_id, service_id, "
                    "max_budget and optional initial_price, initial_sla, engine, "
                    "buyer_strategy, seller_strategy and max_rounds."
                ),
                tags=["negotiation", "marketplace", "bargain"],
                examples=[
                    json.dumps(
                        {
                            "buyer_id": "buyer_1",
                            "seller_id": "seller_1",
                            "service_id": "svc_1",
                            "max_budget": 500,
                            "engine": "strategy",
                            "buyer_strategy": "cooperative",
                            "seller_strategy": "cooperative",
                        }
                    )
                ],
                input_modes=["text", "text/plain", "application/json"],
                output_modes=["text", "text/plain", "application/json"],
            )
        ],
    )


class A2AAuthMiddleware:
    """Require an API key on every A2A request except the public Agent Card.

    Pure ASGI rather than a FastAPI dependency: the SDK registers its own
    routes, so there is no router to hang ``Depends(verify_api_key)`` on.
    Checks run per request, so they follow the live settings like every other
    protected route.
    """

    def __init__(self, app: ASGIApp, *, prefix: str = A2A_PREFIX) -> None:
        self.app = app
        self.prefix = prefix

    def _protected(self, path: str) -> bool:
        return (path == self.prefix or path.startswith(f"{self.prefix}/")) and (
            path != A2A_CARD_URL
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and self._protected(scope.get("path", ""))
            and not headers_authenticate(Headers(scope=scope))
        ):
            response = JSONResponse(
                {"detail": "A valid API key is required. Send it as X-API-Key."},
                status_code=401,
                headers={"WWW-Authenticate": API_KEY_HEADER},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class NegotiationAgentExecutor(AgentExecutor):
    """Runs a negotiation for each A2A task and publishes the outcome.

    The executor is handed two callables that resolve to the app-scoped
    ``DatabaseManager`` and ``ConnectionManager`` at *request* time (after the
    FastAPI lifespan has populated ``app.state``), so A2A and HTTP share one
    database connection and one dashboard event stream.
    """

    def __init__(self, get_db, get_ws_manager) -> None:
        self._get_db = get_db
        self._get_ws_manager = get_ws_manager

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""

        # --- Parse the incoming message into a NegotiationRequest. ---
        payload = self._parse_payload(context)
        if payload is None:
            await self._fail(
                event_queue,
                task_id,
                context_id,
                "Invalid negotiation request: the message must be a JSON-encoded "
                "NegotiationRequest object.",
            )
            return
        try:
            request = NegotiationRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — surface the model's own text
            await self._fail(
                event_queue, task_id, context_id, f"Invalid negotiation request: {exc}"
            )
            return

        # --- Report the task as submitted, then run the shared pipeline. ---
        await event_queue.enqueue_event(
            new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED)
        )
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id, context_id, TaskState.TASK_STATE_WORKING, "Negotiating…"
            )
        )

        db = self._get_db()
        ws_manager = self._get_ws_manager()

        try:
            negotiation_id, initial = await prepare_negotiation(db, request)
        except NegotiationSetupError as exc:
            await self._fail(event_queue, task_id, context_id, exc.detail, submitted=True)
            return

        try:
            summary = await run_negotiation(
                initial, negotiation_id, db, broadcast=ws_manager.broadcast_event
            )
        except Exception as exc:  # noqa: BLE001 — crash -> failed task, not a traceback
            logger.exception("A2A negotiation task %s crashed", task_id)
            await self._fail(
                event_queue, task_id, context_id, f"Negotiation failed: {exc}", submitted=True
            )
            return

        # --- Publish the outcome artifact, then a terminal status. Every
        # negotiation outcome (accepted/rejected/timeout/failed) is a completed
        # *task*: the deal result rides in the artifact. ---
        await event_queue.enqueue_event(
            new_data_artifact_update_event(
                task_id,
                context_id,
                "negotiation_result",
                summary,
                media_type="application/json",
                last_chunk=True,
            )
        )
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id,
                context_id,
                TaskState.TASK_STATE_COMPLETED,
                f"Negotiation {negotiation_id} finished with status {summary['status']}.",
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Mark the task cancelled.

        Stopping the work is the SDK's half: after this returns it cancels the
        task running ``execute``, and ``run_negotiation`` records the
        negotiation as ``failed`` on its way out instead of leaving it active.
        """
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id, context_id, TaskState.TASK_STATE_CANCELED, "Cancelled"
            )
        )

    async def _fail(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        message: str,
        *,
        submitted: bool = False,
    ) -> None:
        """Publish a failed status carrying ``message``.

        The task is created first unless ``submitted`` says ``execute`` already
        did: a second ``new_task`` for an existing task restarts its history.
        """
        if not submitted:
            await event_queue.enqueue_event(
                new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED)
            )
        await event_queue.enqueue_event(
            new_text_status_update_event(task_id, context_id, TaskState.TASK_STATE_FAILED, message)
        )

    @staticmethod
    def _parse_payload(context: RequestContext) -> dict[str, Any] | None:
        """Decode the request from a text or data part of the user message."""
        message = context.message
        if message is None:
            return None

        text = get_message_text(message).strip()
        if text:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return None
            return decoded if isinstance(decoded, dict) else None

        data_parts = get_data_parts(message.parts)
        if data_parts and isinstance(data_parts[0], dict):
            return data_parts[0]
        return None


def add_a2a_to_app(app: FastAPI, *, public_base_url: str = "") -> None:
    """Mount the A2A Agent Card and JSON-RPC endpoints on ``app``.

    Must be called from ``create_app()`` before the app starts serving; the
    executor resolves ``app.state`` lazily at request time.
    """
    card = build_agent_card(public_base_url)
    executor = NegotiationAgentExecutor(
        get_db=lambda: app.state.db,
        get_ws_manager=lambda: app.state.ws_manager,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app.add_middleware(A2AAuthMiddleware)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card, card_url=A2A_CARD_URL),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=A2A_RPC_URL),
    )
