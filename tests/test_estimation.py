"""Strategies see their own limit, and only an *estimate* of the opponent's.

The graph state carries both true limits; the strategy node used to copy both
into ``NegotiationContext``, so every strategy read the other side's secret.
"""

from __future__ import annotations

from agent_bargain.agents.guardrails import GuardedDecision, build_turn_message
from agent_bargain.agents.strategy_node import create_strategy_node
from agent_bargain.engine.negotiation import create_initial_state
from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.estimation import (
    ASSUMED_BUYER_HEADROOM,
    ASSUMED_SELLER_MARGIN,
    estimate_opponent_limit,
    limits_for,
)


class _Spy(NegotiationStrategy):
    name = "spy"

    def __init__(self) -> None:
        self.contexts: list[NegotiationContext] = []

    def decide(self, context: NegotiationContext) -> Decision:
        self.contexts.append(context)
        return Decision(action="counter", price=context.current_price - 10)


def _state(*, max_budget=500.0, min_price=100.0, initial_price=400.0, messages=()):
    state = create_initial_state(
        service_id="s",
        service_title="t",
        buyer_id="b",
        seller_id="se",
        max_budget=max_budget,
        min_price=min_price,
        initial_price=initial_price,
    )
    state["messages"] = list(messages)
    return state


def _turn(role: str, price: float):
    decision = GuardedDecision("counter", price, "Standard", "")
    return build_turn_message(role=role, decision=decision, raw_data={"price": price})


# -- the information boundary -------------------------------------------------


def test_buyer_strategy_never_sees_the_true_floor():
    spies = []
    for true_floor in (10.0, 100.0, 390.0):
        spy = _Spy()
        create_strategy_node("buyer", "max_budget", spy)(_state(min_price=true_floor))
        spies.append(spy.contexts[0])

    # Identical contexts whatever the seller's real floor is.
    assert len({(c.max_budget, c.min_price) for c in spies}) == 1
    assert spies[0].max_budget == 500.0  # own limit is exact
    assert spies[0].own_limit == 500.0


def test_seller_strategy_never_sees_the_true_budget():
    contexts = []
    for true_budget in (150.0, 500.0, 5000.0):
        spy = _Spy()
        state = _state(max_budget=true_budget, messages=[_turn("buyer", 300.0)])
        create_strategy_node("seller", "min_price", spy)(state)
        contexts.append(spy.contexts[0])

    assert len({(c.max_budget, c.min_price) for c in contexts}) == 1
    assert contexts[0].min_price == 100.0
    assert contexts[0].own_limit == 100.0


# -- the estimator ------------------------------------------------------------


def test_buyer_estimate_tracks_the_lowest_ask():
    history = [("seller", 380.0), ("buyer", 200.0), ("seller", 300.0)]
    estimate = estimate_opponent_limit("buyer", initial_price=400.0, price_history=history)
    assert estimate == round(300.0 * (1 - ASSUMED_SELLER_MARGIN), 2)


def test_buyer_estimate_without_asks_uses_the_listing_price():
    estimate = estimate_opponent_limit("buyer", initial_price=400.0, price_history=[])
    assert estimate == round(400.0 * (1 - ASSUMED_SELLER_MARGIN), 2)


def test_seller_estimate_tracks_the_highest_bid():
    history = [("buyer", 300.0), ("seller", 450.0), ("buyer", 420.0)]
    estimate = estimate_opponent_limit("seller", initial_price=400.0, price_history=history)
    assert estimate == round(420.0 * (1 + ASSUMED_BUYER_HEADROOM), 2)


def test_seller_estimate_is_not_dragged_below_the_listing_price_by_a_lowball():
    estimate = estimate_opponent_limit(
        "seller", initial_price=400.0, price_history=[("buyer", 10.0)]
    )
    assert estimate == 400.0


def test_limits_for_puts_the_true_limit_in_the_roles_own_field():
    assert limits_for("buyer", own_limit=500.0, initial_price=400.0, price_history=[]) == (
        500.0,
        estimate_opponent_limit("buyer", initial_price=400.0, price_history=[]),
    )
    assert limits_for("seller", own_limit=100.0, initial_price=400.0, price_history=[])[1] == 100.0
