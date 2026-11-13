"""
Generic agent node factory.

Buyer and Seller nodes share ~90% of their logic. This module extracts the
common pattern into a reusable factory function so each agent file remains
a thin wrapper.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_bargain.agents.guardrails import (
    Role,
    apply_guardrails,
    history_for,
    opponent_of,
    opponent_reputation,
    own_last_offer,
    turn_update,
)
from agent_bargain.llm.provider import get_fallback_llm, get_negotiation_llm
from agent_bargain.llm.schemas import NegotiationDecision

logger = logging.getLogger(__name__)


async def _decide(get_llm: Any, prompt: list[Any]) -> NegotiationDecision:
    """One structured-output call. Client construction is inside the caller's
    guard on purpose: missing/invalid provider config raises right here."""
    llm: BaseChatModel = get_llm()
    return await llm.with_structured_output(NegotiationDecision).ainvoke(prompt)


async def _decide_with_fallback(prompt: list[Any], state: dict[str, Any]) -> NegotiationDecision:
    """Primary LLM, then the fallback LLM, then a reject at the current terms.

    An exception escaping the node would abort the graph and strand the
    negotiation, so every failure degrades to a decision instead.
    """
    try:
        return await _decide(get_negotiation_llm, prompt)
    except Exception:
        logger.warning("Structured output failed, trying fallback LLM.", exc_info=True)
    try:
        return await _decide(get_fallback_llm, prompt)
    except Exception:
        logger.warning("Fallback LLM also failed, defaulting to reject.", exc_info=True)
    return NegotiationDecision(
        action="reject",
        price=float(state.get("current_price", 0.0)),
        sla=str(state.get("current_sla", "")),
        reasoning="Failed to parse decision.",
    )


def create_negotiation_node(
    prompt_template: ChatPromptTemplate,
    role: Role,
    budget_key: str,
):
    """
    Return a LangGraph node function for the given role.

    Args:
        prompt_template: The ChatPromptTemplate for this role.
        role: "buyer" or "seller".
        budget_key: The state key holding the budget constraint
                    ("max_budget" for buyer, "min_price" for seller).
    """
    opponent = opponent_of(role)

    # Async on purpose. A sync node runs in a worker thread, and when the
    # negotiation's wall-clock timeout cancels the graph the thread keeps the
    # LLM request (and its socket) alive until the provider answers. ``ainvoke``
    # is cancelled with the graph instead.
    async def node(state: dict[str, Any]) -> dict[str, Any]:
        # The opponent's turns without their reasoning: an LLM writes its own
        # limit there ("my budget is $500"), and this text is the other side's prompt.
        history_text = "\n".join(history_for(role, state.get("messages", [])))
        rep_score, rep_summary = opponent_reputation(state, role)

        prompt = prompt_template.format_messages(
            service_title=state.get("service_title", "Unknown Service"),
            current_price=state.get("current_price", 0.0),
            current_sla=state.get("current_sla", "Standard"),
            **{budget_key: state.get(budget_key, 0.0)},
            history=history_text or "No history yet. This is the first round.",
            **{
                f"{opponent}_reputation_score": rep_score,
                f"{opponent}_reputation_summary": rep_summary,
            },
        )

        decision = await _decide_with_fallback(prompt, state)
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
        return turn_update(role, guarded)

    node.__name__ = f"{role}_node"
    node.__doc__ = f"LangGraph node for the {role} agent."
    return node
