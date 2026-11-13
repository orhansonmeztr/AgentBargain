"""Strategy-based agent node factory (LLM-free deterministic negotiation).

Creates LangGraph node functions backed by a ``NegotiationStrategy`` instance
instead of an LLM.  Strategies stay stateless: inter-round history is passed
through ``NegotiationContext.metadata["price_history"]``, so reusing the same
instance across rounds *and* across negotiations is safe.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_bargain.agents.guardrails import (
    apply_guardrails,
    history_for,
    opponent_reputation,
    own_last_offer,
    turn_update,
)
from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.estimation import limits_for

logger = logging.getLogger(__name__)


def _price_history(messages: list[Any]) -> list[tuple[str, float]]:
    """Extract ``(role, price)`` pairs from graph messages.

    Strategies receive the negotiation history through
    ``NegotiationContext.metadata["price_history"]`` instead of keeping mutable
    instance state, so a single strategy instance stays stateless.
    """
    history: list[tuple[str, float]] = []
    for msg in messages:
        kwargs = getattr(msg, "additional_kwargs", None) or {}
        raw = kwargs.get("raw_data") or {}
        price = raw.get("price")
        role = kwargs.get("role") or raw.get("role")
        if price is None or role is None:
            continue
        history.append((str(role), float(price)))
    return history


def create_strategy_node(
    role: str,
    budget_key: str,
    strategy_instance: NegotiationStrategy,
):
    """Return a LangGraph node backed by a pre-instantiated strategy (no LLM).

    Args:
        role:              ``"buyer"`` or ``"seller"``.
        budget_key:        State key for the budget constraint
                           (``"max_budget"`` for buyer, ``"min_price"`` for seller).
        strategy_instance: Pre-instantiated strategy. Stateless by contract, so
                           it is safe to reuse across rounds and negotiations.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        from agent_bargain.config import settings

        opponent_score, opponent_summary = opponent_reputation(state, role)

        price_history = _price_history(state.get("messages", []))
        # Only this role's own limit is real; the opponent's is estimated from
        # public information (see strategies/estimation.py). The graph state
        # holds both true limits, so handing them over directly would let every
        # strategy read the other side's secret.
        max_budget, min_price = limits_for(
            role,  # type: ignore[arg-type]
            own_limit=float(state.get(budget_key, 0.0)),
            initial_price=float(state.get("initial_price", 0.0)),
            price_history=price_history,
        )
        context = NegotiationContext(
            role=role,  # type: ignore[arg-type]
            max_budget=max_budget,
            min_price=min_price,
            current_price=float(state.get("current_price", 0.0)),
            current_sla=str(state.get("current_sla", "Standard")),
            initial_price=float(state.get("initial_price", 0.0)),
            round=int(state.get("round", 0)),
            max_rounds=int(state.get("max_rounds", settings.max_negotiation_rounds)),
            opponent_reputation=opponent_score,
            opponent_reputation_summary=opponent_summary,
            message_history=history_for(role, state.get("messages", [])),
            metadata={"price_history": price_history},
        )

        try:
            decision: Decision = strategy_instance.decide(context)
        except Exception:
            logger.exception(
                "Strategy %r raised an error; defaulting to reject.", strategy_instance.name
            )
            decision = Decision(
                action="reject",
                price=context.current_price,
                sla=context.current_sla,
                reasoning="Strategy error — rejecting.",
            )

        guarded = apply_guardrails(
            action=decision.action,
            price=decision.price,
            sla=decision.sla,
            reasoning=decision.reasoning,
            role=role,
            limit=float(state.get(budget_key, 0.0)),
            current_price=float(state.get("current_price", 0.0)),
            own_last_offer=own_last_offer(role, state.get("messages", [])),
        )

        return turn_update(role, guarded)  # type: ignore[arg-type]

    node.__name__ = f"{role}_strategy_node[{strategy_instance.name}]"
    node.__doc__ = (
        f"LangGraph node for {role} using the '{strategy_instance.name}' strategy (no LLM)."
    )
    return node
