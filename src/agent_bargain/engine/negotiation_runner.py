"""Reusable negotiation execution core.

The cost of running a negotiation — loading reputation, driving the LangGraph,
persisting the record/transcript/agreement, and the post-trade evaluation — is
shared by the HTTP API and the A2A server. This module is the single copy; the
two surfaces only differ in how they surface the outcome:
``api/routes/negotiations.py`` broadcasts WebSocket events and returns a
``negotiation_id`` immediately, while ``protocols/a2a_server.py`` publishes A2A
events and returns the task's final artifact.

``prepare_negotiation`` raises a :class:`~agent_bargain.engine.errors.NegotiationSetupError`
on validation failure so both callers map the error to their own transport (an
HTTP status, or a failed A2A task with the same detail text).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from agent_bargain.agents.evaluator import evaluate_agreement
from agent_bargain.agents.guardrails import history_for, public_content
from agent_bargain.config import settings
from agent_bargain.engine.errors import (
    OpeningBelowFloorError,
    SellerMismatchError,
    ServiceNotRegisteredError,
)
from agent_bargain.engine.negotiation import (
    create_initial_state,
    create_negotiation_graph,
    create_strategy_graph,
)
from agent_bargain.engine.reputation import (
    DEFAULT_REPUTATION_SUMMARY,
    ReputationEngine,
)
from agent_bargain.models.negotiation import (
    Agreement,
    NegotiationMessage,
    NegotiationRecord,
    NegotiationRequest,
    NegotiationStatus,
)
from agent_bargain.models.reputation import FeedbackScore
from agent_bargain.storage.database import DatabaseManager, select_representative_feedbacks

logger = logging.getLogger(__name__)

#: A viewer that is neither party, so :func:`history_for` shows every turn's
#: public form.
EVALUATOR_VIEWER = "evaluator"


def _build_reputation_summary(feedbacks: list) -> str:
    """Format representative feedbacks into a readable summary for LLM prompts."""
    if not feedbacks:
        return DEFAULT_REPUTATION_SUMMARY

    lines = [f"  - Score: {fb.score}/5.0 | {fb.comments or 'No comment'}" for fb in feedbacks]
    return "\n".join(lines)


def _infer_role(content: str) -> str:
    """Last-resort role inference from message label text."""
    if "BUYER:" in content:
        return "buyer"
    if "SELLER:" in content:
        return "seller"
    return "system"


def _transcript(graph_messages: list[Any], negotiation_id: str) -> list[NegotiationMessage]:
    """Persistable rows for the graph's messages, in turn order."""
    messages: list[NegotiationMessage] = []
    for msg in graph_messages:
        raw_data = msg.additional_kwargs.get("raw_data")
        role = msg.additional_kwargs.get("role") or _infer_role(msg.content)
        ts_str = msg.additional_kwargs.get("timestamp")
        msg_kwargs: dict[str, Any] = {
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "negotiation_id": negotiation_id,
            "role": role,
            "content": msg.content,
            "raw_data": json.dumps(raw_data) if raw_data else None,
        }
        if ts_str:
            msg_kwargs["timestamp"] = datetime.fromisoformat(ts_str)
        messages.append(NegotiationMessage(**msg_kwargs))
    return messages


async def prepare_negotiation(
    db: DatabaseManager,
    request: NegotiationRequest,
) -> tuple[str, dict[str, Any]]:
    """Validate a negotiation request and return ``(negotiation_id, initial_state)``.

    Takes the shared :class:`NegotiationRequest` that both the HTTP route and
    the A2A executor have already validated, so the two surfaces cannot drift
    in how they map it. The service must already be registered by its seller;
    the seller's floor price is read from that listing rather than supplied by
    the caller. Raises a :class:`~agent_bargain.engine.errors.NegotiationSetupError`
    subclass (carrying its 404/409/422 status) so each surface maps it to its
    own transport.
    """
    buyer_id, seller_id, service_id = request.buyer_id, request.seller_id, request.service_id
    max_budget, initial_price = request.max_budget, request.initial_price
    negotiation_id = f"neg_{uuid.uuid4().hex[:8]}"

    service = await db.get_service(service_id)
    if service is None:
        raise ServiceNotRegisteredError(
            f"Service '{service_id}' is not registered. "
            "POST /api/v1/marketplace/services first — the seller's floor "
            "price lives on the listing, so it cannot be supplied here."
        )
    if service.seller_id != seller_id:
        raise SellerMismatchError(
            f"Service '{service_id}' belongs to seller '{service.seller_id}', not '{seller_id}'."
        )

    seller_floor = service.floor_price
    opening_price = initial_price if initial_price is not None else service.base_price
    if opening_price < seller_floor:
        raise OpeningBelowFloorError(
            f"initial_price {opening_price} is below the seller's registered "
            f"floor price for '{service_id}'."
        )
    opening_sla = request.initial_sla if request.initial_sla is not None else service.sla_terms

    # Foreign keys are enforced; agents are inserted only if absent, so a real
    # agent's budget/capacity is never clobbered.
    await db.ensure_agent(buyer_id)
    await db.ensure_agent(seller_id)

    initial = create_initial_state(
        service_id=service_id,
        service_title=service.title,
        buyer_id=buyer_id,
        seller_id=seller_id,
        max_budget=max_budget,
        min_price=seller_floor,
        initial_price=opening_price,
        initial_sla=opening_sla,
        engine=request.engine,
        buyer_strategy=request.buyer_strategy or "",
        seller_strategy=request.seller_strategy or "",
        max_rounds=request.max_rounds,
    )

    # Persist an "active" record up-front so status queries succeed while the
    # negotiation is still running.
    try:
        await db.save_negotiation(
            NegotiationRecord(
                negotiation_id=negotiation_id,
                buyer_id=buyer_id,
                seller_id=seller_id,
                service_id=service_id,
                max_budget=max_budget,
                min_price=seller_floor,
                initial_price=opening_price,
                status=NegotiationStatus.active,
                rounds=0,
                max_rounds=initial["max_rounds"],
            )
        )
    except Exception:
        logger.exception(
            "Failed to persist initial 'active' record for %s — status queries may 404",
            negotiation_id,
        )

    return negotiation_id, initial


async def run_negotiation(
    state: dict[str, Any],
    negotiation_id: str,
    db: DatabaseManager,
    *,
    broadcast: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run the negotiation graph, persist the outcome, and return a summary.

    ``db`` is the application's shared connection (``app.state.db``); the caller
    owns its lifecycle. ``broadcast`` receives already-shaped event dicts (the
    same ones the WebSocket layer serialises) and may be ``None`` to skip
    broadcasting entirely.
    """
    buyer_id = state.get("buyer_id", "unknown_buyer")
    seller_id = state.get("seller_id", "unknown_seller")
    service_id = state.get("service_id", "unknown_service")

    async def _emit(event: dict[str, Any]) -> None:
        if broadcast is not None:
            await broadcast(event)

    # --- Load reputation context ---
    try:
        rep_engine = ReputationEngine()
        all_buyer_feedbacks = await db.get_feedbacks_for_agent(buyer_id)
        all_seller_feedbacks = await db.get_feedbacks_for_agent(seller_id)
        state["buyer_reputation_score"] = rep_engine.score_or_neutral(all_buyer_feedbacks)
        state["buyer_reputation_summary"] = _build_reputation_summary(
            select_representative_feedbacks(all_buyer_feedbacks)
        )
        state["seller_reputation_score"] = rep_engine.score_or_neutral(all_seller_feedbacks)
        state["seller_reputation_summary"] = _build_reputation_summary(
            select_representative_feedbacks(all_seller_feedbacks)
        )
        logger.info(
            "Reputation loaded — buyer %s: %.2f, seller %s: %.2f",
            buyer_id,
            state["buyer_reputation_score"],
            seller_id,
            state["seller_reputation_score"],
        )
    except Exception:
        logger.exception("Failed to load reputation data")

    # --- Run the negotiation graph ---
    final_state = dict(state)

    def _build_graph():
        if state.get("engine", "llm") == "strategy":
            logger.info(
                "Negotiation %s running strategy engine: buyer=%s, seller=%s",
                negotiation_id,
                state.get("buyer_strategy"),
                state.get("seller_strategy"),
            )
            return create_strategy_graph(
                state.get("buyer_strategy", ""),
                state.get("seller_strategy", ""),
            )
        return create_negotiation_graph()

    async def _stream_graph() -> None:
        # Built in here, inside the guarded block below: a failure to build
        # (an unregistered strategy, a broken constructor) used to escape
        # before any handler and leave the row "active" until a restart.
        graph = _build_graph()
        async for step_event in graph.astream(state):
            for node_name, node_state in step_event.items():
                for k, v in node_state.items():
                    if k == "messages":
                        final_state["messages"] = final_state.get("messages", []) + v
                    else:
                        final_state[k] = v

                if "messages" in node_state and node_state["messages"]:
                    last_msg = node_state["messages"][-1]
                    raw = last_msg.additional_kwargs.get("raw_data") or {}
                    await _emit(
                        {
                            "type": "agent_activity",
                            "negotiation_id": negotiation_id,
                            "agent_id": node_state.get("last_actor", node_name),
                            "role": last_msg.additional_kwargs.get("role", node_name),
                            "action": raw.get("action", node_state.get("last_action", "")),
                            "price": raw.get("price"),
                            "sla": raw.get("sla"),
                            "reasoning": raw.get("reasoning", ""),
                            "content": last_msg.content,
                            # Stripped of reasoning; what non-admin WebSocket
                            # clients receive instead of ``content``.
                            "public_content": public_content(last_msg),
                            "round": final_state.get("round", 0),
                            "max_rounds": final_state.get("max_rounds"),
                        }
                    )

    def _record(status: str) -> NegotiationRecord:
        """The persisted row. Request terms are always repeated: the save is an
        upsert, so leaving them out would NULL them on the existing row.

        ``final_price``/``final_sla`` are set only for an accepted deal. On a
        reject or timeout ``current_price`` is merely the last offer nobody
        took, and storing it as a "final price" misreported it everywhere."""
        outcome = status == NegotiationStatus.accepted
        return NegotiationRecord(
            negotiation_id=negotiation_id,
            buyer_id=buyer_id,
            seller_id=seller_id,
            service_id=service_id,
            max_budget=state.get("max_budget"),
            min_price=state.get("min_price"),
            initial_price=state.get("initial_price"),
            status=status,
            rounds=final_state.get("round", 0),
            max_rounds=state.get("max_rounds"),
            final_price=final_state.get("current_price") if outcome else None,
            final_sla=final_state.get("current_sla") if outcome else None,
        )

    try:
        await asyncio.wait_for(_stream_graph(), timeout=settings.negotiation_timeout_seconds)
    except TimeoutError:
        logger.warning(
            "Negotiation %s timed out after %ss (wall clock)",
            negotiation_id,
            settings.negotiation_timeout_seconds,
        )
        if final_state.get("status", "active") == "active":
            final_state["status"] = "timeout"
        else:
            logger.info(
                "Negotiation %s kept its pre-timeout status %r",
                negotiation_id,
                final_state.get("status"),
            )
    except Exception:
        # Any other crash inside the graph must still reach persistence below:
        # left unhandled it skipped the final save and the row stayed "active"
        # — indistinguishable from a live negotiation — until the next restart.
        logger.exception("Negotiation %s crashed inside the graph", negotiation_id)
        final_state["status"] = NegotiationStatus.failed
    except asyncio.CancelledError:
        # Cancelled from outside — an A2A ``tasks/cancel``, or server shutdown.
        # Record a terminal status before letting the cancellation propagate,
        # or the row stays "active" until the next restart's orphan sweep.
        logger.warning("Negotiation %s was cancelled; recording it as failed", negotiation_id)
        try:
            await db.save_negotiation(_record(NegotiationStatus.failed))
        except Exception:
            logger.exception("Could not record cancelled negotiation %s", negotiation_id)
        # Tell clients as well: without this event the dashboard kept showing
        # the negotiation as live until the page was reloaded. Best-effort —
        # a broken socket must not replace the cancellation with another error.
        try:
            await _emit(
                {
                    "type": "negotiation_update",
                    "negotiation_id": negotiation_id,
                    "status": str(NegotiationStatus.failed),
                    "rounds": final_state.get("round", 0),
                    "final_price": None,
                }
            )
        except Exception:
            logger.exception("Could not broadcast cancelled negotiation %s", negotiation_id)
        raise

    status = NegotiationStatus(final_state.get("status") or NegotiationStatus.failed)
    rounds = final_state.get("round", 0)
    accepted = status == NegotiationStatus.accepted
    final_price = final_state.get("current_price") if accepted else None
    final_sla = final_state.get("current_sla") if accepted else None

    summary: dict[str, Any] = {
        "negotiation_id": negotiation_id,
        "status": str(status),
        "rounds": rounds,
        "final_price": final_price,
        "final_sla": final_sla,
        "buyer_id": buyer_id,
        "seller_id": seller_id,
        "service_id": service_id,
    }

    # --- Persist the outcome BEFORE evaluating it ---
    # Row, transcript and agreement are one transaction. Written separately, a
    # failing transcript insert left an "accepted" row with no agreement.
    agreement: Agreement | None = None
    try:
        messages = _transcript(final_state.get("messages", []), negotiation_id)
        if accepted:
            agreement = Agreement(
                agreement_id=f"agr_{uuid.uuid4().hex[:8]}",
                buyer_id=buyer_id,
                seller_id=seller_id,
                service_id=service_id,
                final_price=final_price if final_price is not None else 0.0,
                final_sla=final_sla or "",
            )
        await db.record_outcome(_record(status), messages, agreement)
    except Exception:
        logger.exception("Failed to save negotiation outcome — attempting failed-status fallback")
        try:
            await db.save_negotiation(_record(NegotiationStatus.failed))
        except Exception:
            logger.exception(
                "Failed-status fallback also failed — negotiation %s lost", negotiation_id
            )
        # Tell clients too, or the dashboard shows the negotiation as live forever.
        await _emit(
            {
                "type": "negotiation_update",
                "negotiation_id": negotiation_id,
                "status": str(NegotiationStatus.failed),
                "rounds": rounds,
                "final_price": None,
            }
        )
        return {**summary, "status": "failed", "final_price": None, "final_sla": None}

    await _emit(
        {
            "type": "negotiation_update",
            "negotiation_id": negotiation_id,
            "status": str(status),
            "rounds": rounds,
            "final_price": final_price,
        }
    )

    # --- Post-trade evaluation (only when a deal was struck) ---
    if agreement is None:
        return summary

    evaluation = await evaluate_agreement(
        final_price=final_price,
        final_sla=final_sla,
        # The public transcript: actions, prices, SLAs — no reasoning. Each
        # party's reasoning routinely states its own limit, and the evaluator's
        # comments become the reputation summary the *next* counterparty reads.
        history=history_for(EVALUATOR_VIEWER, final_state.get("messages", [])),
    )
    if evaluation is None:
        return summary

    feedbacks = [
        FeedbackScore(
            feedback_id=f"fb_{uuid.uuid4().hex[:8]}",
            agreement_id=agreement.agreement_id,
            evaluator_id="system_evaluator",
            target_id=target_id,
            score=float(score),
            comments=comments,
            evaluator_model=settings.evaluation_model,
        )
        for target_id, score, comments in (
            (seller_id, evaluation.seller_score, evaluation.seller_comments),
            (buyer_id, evaluation.buyer_score, evaluation.buyer_comments),
        )
    ]
    try:
        # Feedback and the reputation it implies land together or not at all.
        await db.record_evaluation(feedbacks, ReputationEngine().score_or_neutral)
    except Exception:
        logger.exception("Failed to persist evaluation feedback for %s", negotiation_id)
        return summary

    await _emit(
        {
            "type": "evaluation_update",
            "negotiation_id": negotiation_id,
            "seller_score": float(evaluation.seller_score),
            "seller_comments": evaluation.seller_comments,
            "buyer_score": float(evaluation.buyer_score),
            "buyer_comments": evaluation.buyer_comments,
        }
    )

    return summary
