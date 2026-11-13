"""LangGraph negotiation state machine."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_bargain.agents.buyer import buyer_node
from agent_bargain.agents.seller import seller_node
from agent_bargain.config import settings
from agent_bargain.engine.reputation import DEFAULT_REPUTATION_SUMMARY, NEUTRAL_REPUTATION_SCORE


class NegotiationState(TypedDict):
    """The state dictionary for the negotiation LangGraph."""

    service_id: str
    service_title: str
    buyer_id: str
    seller_id: str
    max_budget: float
    min_price: float
    initial_price: float
    current_price: float
    current_sla: str
    messages: Annotated[list[Any], operator.add]
    round: int
    max_rounds: int  # per-negotiation round limit (defaults to settings.max_negotiation_rounds)
    status: str  # "active", "accepted", "rejected", "timeout"
    last_action: str
    last_actor: str
    # Reputation context
    buyer_reputation_score: float
    buyer_reputation_summary: str
    seller_reputation_score: float
    seller_reputation_summary: str
    # Engine selection
    engine: str  # "llm" | "strategy"
    buyer_strategy: str  # strategy name when engine="strategy", "" otherwise
    seller_strategy: str  # strategy name when engine="strategy", "" otherwise


def create_initial_state(
    *,
    service_id: str,
    service_title: str,
    buyer_id: str,
    seller_id: str,
    max_budget: float,
    min_price: float,
    initial_price: float,
    initial_sla: str = "Standard",
    buyer_reputation_score: float = NEUTRAL_REPUTATION_SCORE,
    buyer_reputation_summary: str = DEFAULT_REPUTATION_SUMMARY,
    seller_reputation_score: float = NEUTRAL_REPUTATION_SCORE,
    seller_reputation_summary: str = DEFAULT_REPUTATION_SUMMARY,
    engine: str = "llm",
    buyer_strategy: str = "",
    seller_strategy: str = "",
    max_rounds: int | None = None,
) -> NegotiationState:
    """Create a fully initialized state dict with safe defaults for all fields."""
    return NegotiationState(
        service_id=service_id,
        service_title=service_title,
        buyer_id=buyer_id,
        seller_id=seller_id,
        max_budget=max_budget,
        min_price=min_price,
        initial_price=initial_price,
        current_price=initial_price,
        current_sla=initial_sla,
        messages=[],
        round=0,
        max_rounds=max_rounds if max_rounds is not None else settings.max_negotiation_rounds,
        status="active",
        last_action="",
        last_actor="",
        buyer_reputation_score=buyer_reputation_score,
        buyer_reputation_summary=buyer_reputation_summary,
        seller_reputation_score=seller_reputation_score,
        seller_reputation_summary=seller_reputation_summary,
        engine=engine,
        buyer_strategy=buyer_strategy,
        seller_strategy=seller_strategy,
    )


def _increment_round(state: NegotiationState) -> dict[str, Any]:
    """Increment the round counter after a full buyer->seller exchange."""
    return {"round": state.get("round", 0) + 1}


def should_continue(state: NegotiationState) -> str:
    """Determine the next node based on the last action."""
    action = state.get("last_action", "").lower()

    if action == "accept":
        return "accepted"
    if action == "reject":
        return "rejected"

    # Use per-negotiation max_rounds; fall back to global setting.
    max_rounds = state.get("max_rounds", settings.max_negotiation_rounds)
    if state.get("round", 0) >= max_rounds:
        return "timeout"

    # Counter: hand over to the other party
    if state.get("last_actor") == "buyer":
        return "seller"
    return "buyer"


def _handle_agreement(state: NegotiationState) -> dict[str, Any]:
    return {"status": "accepted"}


def _handle_rejection(state: NegotiationState) -> dict[str, Any]:
    return {"status": "rejected"}


def _handle_timeout(state: NegotiationState) -> dict[str, Any]:
    return {"status": "timeout"}


def _build_workflow(buyer_node_fn: Any, seller_node_fn: Any) -> StateGraph:
    """Assemble the shared graph topology with the given buyer/seller nodes.

    The graph models *bargaining only*. Post-trade evaluation used to be a node
    on the accepted path (``accepted -> evaluator -> END``), which meant the
    terminal status could not be persisted until a 15-40s LLM call finished —
    a deal struck in 50ms stayed "active" to every client for that long. The
    evaluation now runs after the outcome is persisted; see
    ``engine/negotiation_runner.py``.
    """
    workflow = StateGraph(NegotiationState)

    # Agent nodes
    workflow.add_node("buyer", buyer_node_fn)
    workflow.add_node("seller", seller_node_fn)
    workflow.add_node("round_counter", _increment_round)

    # Terminal nodes
    workflow.add_node("accepted", _handle_agreement)
    workflow.add_node("rejected", _handle_rejection)
    workflow.add_node("timeout", _handle_timeout)

    # Entry point
    workflow.set_entry_point("buyer")

    _CONDITIONAL = {
        "seller": "seller",
        "accepted": "accepted",
        "rejected": "rejected",
        "timeout": "timeout",
    }

    workflow.add_conditional_edges("buyer", should_continue, _CONDITIONAL)
    workflow.add_edge("seller", "round_counter")
    workflow.add_conditional_edges(
        "round_counter",
        should_continue,
        {"buyer": "buyer", **_CONDITIONAL},
    )

    # Every terminal state ends the graph directly.
    workflow.add_edge("accepted", END)
    workflow.add_edge("rejected", END)
    workflow.add_edge("timeout", END)

    return workflow


def create_negotiation_graph() -> CompiledStateGraph:
    """Create and compile the LLM-backed negotiation state machine.

    Flow::

        buyer -> seller -> round_counter -> (continue?)
        ...
        accepted -> END
        rejected -> END
        timeout  -> END

    Post-trade evaluation is not part of this graph; the caller runs it after
    persisting the outcome.
    """
    return _build_workflow(buyer_node, seller_node).compile()


def create_strategy_graph(buyer_strategy: str, seller_strategy: str) -> CompiledStateGraph:
    """Create a compiled strategy-only graph (no LLM for negotiation turns).

    Built-in strategies are stateless (history arrives through
    ``NegotiationContext.metadata["price_history"]``), but a fresh instance per
    negotiation keeps third-party strategies that *do* hold state from leaking
    it across negotiations. Within one negotiation the instance is reused.

    Post-trade evaluation still uses an LLM, but runs outside this graph.

    Args:
        buyer_strategy:  Registered strategy name for the buyer role.
        seller_strategy: Registered strategy name for the seller role.

    Raises:
        UnknownStrategyError: If either strategy name is not registered.
    """
    from agent_bargain.agents.strategy_node import create_strategy_node
    from agent_bargain.strategies.registry import StrategyRegistry

    buyer_instance = StrategyRegistry.instantiate(buyer_strategy)
    seller_instance = StrategyRegistry.instantiate(seller_strategy)

    buyer_node_fn = create_strategy_node("buyer", "max_budget", buyer_instance)
    seller_node_fn = create_strategy_node("seller", "min_price", seller_instance)

    return _build_workflow(buyer_node_fn, seller_node_fn).compile()
