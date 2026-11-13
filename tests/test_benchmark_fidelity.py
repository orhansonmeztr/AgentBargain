"""The benchmark runner must simulate the same agent the graph runs.

``README.md`` publishes a leaderboard produced by ``BenchmarkRunner``. If the
runner feeds strategies a different context than ``create_strategy_node`` does
— or skips the guardrails — the leaderboard ranks an agent that cannot occur
in production. These tests pin the two paths together.
"""

from __future__ import annotations

import pytest

from agent_bargain.benchmarks.runner import BenchmarkRunner
from agent_bargain.benchmarks.scenarios import SCENARIOS, get_scenario
from agent_bargain.engine.negotiation import create_initial_state, create_strategy_graph

STRATEGY_PAIRS = [
    ("cooperative", "cooperative"),
    ("cooperative", "aggressive"),
    ("tit_for_tat", "tit_for_tat"),
    ("tit_for_tat", "anchoring"),
    ("aggressive", "concession"),
    ("anchoring", "cooperative"),
    ("concession", "tit_for_tat"),
]


def _graph_prices(messages) -> list[float]:
    """Prices in turn order, read the same way analytics reads them."""
    return [
        float(msg.additional_kwargs["raw_data"]["price"])
        for msg in messages
        if msg.additional_kwargs.get("raw_data")
    ]


def _run_graph(buyer_name: str, seller_name: str, scenario) -> dict:
    """Run the real LangGraph strategy engine over a scenario's parameters."""
    state = create_initial_state(
        service_id="svc_bench",
        service_title="Benchmark Service",
        buyer_id="buyer_bench",
        seller_id="seller_bench",
        max_budget=scenario.buyer_max,
        min_price=scenario.seller_min,
        initial_price=scenario.initial_price,
        engine="strategy",
        buyer_strategy=buyer_name,
        seller_strategy=seller_name,
        max_rounds=scenario.max_rounds,
    )
    graph = create_strategy_graph(buyer_name, seller_name)
    return graph.invoke(state)


# No evaluator stub is needed here: post-trade evaluation is no longer a graph
# node, so running the bargaining graph makes no LLM call at all. That is the
# point of the change — see test_negotiation.test_graph_models_bargaining_only.


@pytest.mark.parametrize(("buyer_name", "seller_name"), STRATEGY_PAIRS)
def test_benchmark_trajectory_matches_graph(buyer_name, seller_name):
    """Same strategies + same scenario => identical price trajectory."""
    scenario = get_scenario("balanced")

    final_state = _run_graph(buyer_name, seller_name, scenario)
    graph_prices = _graph_prices(final_state["messages"])

    match = BenchmarkRunner()._run_match(buyer_name, seller_name, scenario)

    # The benchmark seeds its trajectory with the opening price, which the
    # graph holds in state rather than in a message.
    assert match.report.concession.price_trajectory == [scenario.initial_price, *graph_prices]


@pytest.mark.parametrize(("buyer_name", "seller_name"), STRATEGY_PAIRS)
def test_benchmark_outcome_and_rounds_match_graph(buyer_name, seller_name):
    """Outcome, final price and round count must agree with the graph."""
    scenario = get_scenario("balanced")

    final_state = _run_graph(buyer_name, seller_name, scenario)
    match = BenchmarkRunner()._run_match(buyer_name, seller_name, scenario)

    assert match.outcome == final_state["status"]
    assert match.rounds_used == final_state["round"]
    if match.outcome == "accepted":
        assert match.final_price == pytest.approx(final_state["current_price"])


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_benchmark_matches_graph_across_all_scenarios(scenario):
    """Fidelity must hold for every market condition, not just "balanced"."""
    buyer_name, seller_name = "tit_for_tat", "aggressive"

    final_state = _run_graph(buyer_name, seller_name, scenario)
    match = BenchmarkRunner()._run_match(buyer_name, seller_name, scenario)

    assert match.outcome == final_state["status"]
    assert match.rounds_used == final_state["round"]
    assert match.report.concession.price_trajectory == [
        scenario.initial_price,
        *_graph_prices(final_state["messages"]),
    ]


def test_guardrails_apply_in_benchmarks():
    """A counter that echoes the price on the table settles, as in the graph.

    The runner used to take ``Decision.price`` verbatim, so a strategy that
    echoed the current price would spin — something the node never allows.
    """
    from agent_bargain.benchmarks.scenarios import Scenario
    from agent_bargain.strategies.base import Decision, NegotiationStrategy
    from agent_bargain.strategies.registry import StrategyRegistry

    class _StubbornStrategy(NegotiationStrategy):
        """Always counters with exactly the price it was shown."""

        description = "test-only: never moves"

        def decide(self, context):
            return Decision(
                action="counter",
                price=context.current_price,
                sla=context.current_sla,
                reasoning="stubborn",
            )

    StrategyRegistry.register("test_stubborn", _StubbornStrategy)
    try:
        scenario = Scenario(
            name="stubborn",
            description="test",
            buyer_max=500.0,
            seller_min=100.0,
            initial_price=400.0,
            max_rounds=4,
        )
        match = BenchmarkRunner()._run_match("test_stubborn", "test_stubborn", scenario)

        # Offering exactly the asking price is agreeing to it: without the
        # guardrail the runner would record a counter and spin to timeout.
        assert match.outcome == "accepted"
        assert match.final_price == scenario.initial_price
        assert match.rounds_used == 0
    finally:
        from agent_bargain.strategies import registry

        registry._REGISTRY.pop("test_stubborn", None)
