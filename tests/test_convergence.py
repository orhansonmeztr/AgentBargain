"""Offers converge between the two openings instead of ratcheting past them.

Every built-in strategy used to compute its next counter from
``context.current_price`` — the *opponent's* latest offer — so a buyer facing a
476 ask "conceded" to 484, the seller conceded from 484 back to 476, and the
deal closed above the seller's own opening. Strategies now concede from their
own last offer, and the turn guardrail turns any counter that meets the price
on the table into an accept.
"""

from __future__ import annotations

import itertools

import pytest

import agent_bargain.strategies  # noqa: F401  (registers the built-ins)
from agent_bargain.benchmarks.runner import BenchmarkRunner
from agent_bargain.benchmarks.scenarios import SCENARIOS, Scenario
from agent_bargain.strategies.base import NegotiationContext
from agent_bargain.strategies.registry import StrategyRegistry

#: The scenario from the LangSmith trace that exposed the bug.
TRACE_SCENARIO = Scenario(
    name="trace",
    description="aggressive vs aggressive trace",
    buyer_max=500.0,
    seller_min=100.0,
    initial_price=400.0,
    max_rounds=10,
)

_PAIRS = list(itertools.product(StrategyRegistry.names(), repeat=2))
_SCENARIOS = [*SCENARIOS, TRACE_SCENARIO]


def _turns(scenario: Scenario, buyer: str, seller: str):
    match = BenchmarkRunner()._run_match(buyer, seller, scenario)
    prices = match.report.concession.price_trajectory[1:]  # drop the listing price
    # Turns strictly alternate, buyer first.
    roles = ["buyer" if i % 2 == 0 else "seller" for i in range(len(prices))]
    turns = list(zip(roles, prices, strict=True))
    # Only the last turn can settle; everything before it was a counter.
    counters = turns if match.outcome == "timeout" else turns[:-1]
    return match, counters


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize(("buyer", "seller"), _PAIRS, ids=lambda n: n)
def test_counters_never_cross_and_never_retreat(scenario, buyer, seller):
    match, counters = _turns(scenario, buyer, seller)

    on_table = scenario.initial_price
    last_own: dict[str, float] = {}
    for role, price in counters:
        if role == "buyer":
            assert price < on_table, f"buyer bid {price} at or above the ask {on_table}"
            assert price >= last_own.get("buyer", price), "buyer retreated"
        else:
            assert price > on_table, f"seller asked {price} at or below the bid {on_table}"
            assert price <= last_own.get("seller", price), "seller retreated"
        last_own[role] = price
        on_table = price

    if match.outcome == "accepted":
        bids = [p for r, p in counters if r == "buyer"]
        asks = [scenario.initial_price, *(p for r, p in counters if r == "seller")]
        lowest = min(bids, default=match.final_price)
        assert lowest <= match.final_price <= max(asks)


def test_trace_scenario_settles_between_the_openings():
    match, counters = _turns(TRACE_SCENARIO, "aggressive", "aggressive")
    buyer_open = counters[0][1]
    seller_open = counters[1][1]

    assert match.outcome == "accepted"
    assert buyer_open < match.final_price < seller_open


# -- context helpers ----------------------------------------------------------


def _context(role, *, history=(), current_price=380.0):
    return NegotiationContext(
        role=role,
        max_budget=500.0,
        min_price=240.0 if role == "buyer" else 100.0,
        current_price=current_price,
        current_sla="Standard",
        initial_price=400.0,
        round=1,
        max_rounds=10,
        opponent_reputation=3.0,
        opponent_reputation_summary="",
        metadata={"price_history": list(history)},
    )


def test_own_position_is_the_roles_own_last_offer():
    history = [("buyer", 250.0), ("seller", 380.0)]
    assert _context("buyer", history=history).own_position == 250.0
    assert _context("seller", history=history).own_position == 380.0


def test_own_position_before_a_first_offer():
    # A buyer starts from its estimate of the seller's floor; a seller from
    # the listing price, its own public ask.
    assert _context("buyer").own_position == 240.0
    assert _context("seller", history=[("buyer", 200.0)]).own_position == 400.0


def test_concede_moves_from_own_offer_not_the_opponents():
    ctx = _context("buyer", history=[("buyer", 250.0), ("seller", 380.0)])
    assert ctx.concede(10.0) == 260.0  # not 390


def test_concede_never_retreats_or_passes_own_limit():
    ctx = _context("buyer", history=[("buyer", 250.0), ("seller", 380.0)])
    assert ctx.concede(-50.0) == 250.0
    assert ctx.concede(10_000.0) == 500.0

    seller = _context("seller", history=[("buyer", 200.0), ("seller", 380.0)])
    assert seller.concede(10.0) == 370.0
    assert seller.concede(10_000.0) == 100.0


# -- deadline and acceptance --------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize(("buyer", "seller"), _PAIRS, ids=lambda n: n)
def test_every_scenario_with_a_real_zopa_closes(scenario, buyer, seller):
    """Both parties reach their own limit by the last round, so none time out apart.

    Concessions used to stall: in ``tight_margin`` and ``seller_advantage`` the
    parties edged toward each other and ran out of rounds with a deal possible.
    """
    match = BenchmarkRunner()._run_match(buyer, seller, scenario)
    if scenario.buyer_max >= scenario.seller_min:
        assert match.outcome == "accepted"
        assert scenario.seller_min <= match.final_price <= scenario.buyer_max
    else:
        assert match.outcome != "accepted"


def _deadline_context(role, *, round_, max_rounds=5, history=()):
    return NegotiationContext(
        role=role,
        max_budget=500.0,
        min_price=100.0,
        current_price=300.0,
        current_sla="Standard",
        initial_price=400.0,
        round=round_,
        max_rounds=max_rounds,
        opponent_reputation=3.0,
        opponent_reputation_summary="",
        metadata={"price_history": list(history)},
    )


def test_deadline_price_runs_from_the_opening_to_the_own_limit():
    history = [("buyer", 200.0), ("seller", 450.0)]
    assert _deadline_context("buyer", round_=0).deadline_price(2.0) is None
    mid = _deadline_context("buyer", round_=2, history=history).deadline_price(1.0)
    assert mid == 350.0  # halfway from the 200 opening to the 500 budget
    last = _deadline_context("buyer", round_=4, history=history).deadline_price(4.0)
    assert last == 500.0
    seller_last = _deadline_context("seller", round_=4, history=history).deadline_price(4.0)
    assert seller_last == 100.0


def test_firmer_curves_concede_later():
    history = [("buyer", 200.0), ("seller", 450.0)]
    ctx = _deadline_context("buyer", round_=2, history=history)
    assert ctx.deadline_price(4.0) < ctx.deadline_price(1.0) < ctx.deadline_price(0.5)


def test_one_round_negotiation_goes_straight_to_the_limit():
    assert _deadline_context("buyer", round_=0, max_rounds=1).deadline_price(3.0) == 500.0


def test_acceptance_compares_the_table_with_the_next_counter():
    buyer = _deadline_context("buyer", round_=1)  # 300 on the table
    assert buyer.offer_is_no_worse_than(300.0)
    assert not buyer.offer_is_no_worse_than(299.0)
    seller = _deadline_context("seller", round_=1)
    assert seller.offer_is_no_worse_than(300.0)
    assert not seller.offer_is_no_worse_than(301.0)


def test_anchoring_seller_no_longer_accepts_a_lowball_just_above_its_floor():
    # It opened at 505; 164 is above 1.2x its 100 floor, which used to suffice.
    ctx = NegotiationContext(
        role="seller",
        max_budget=400.0,
        min_price=100.0,
        current_price=164.0,
        current_sla="Standard",
        initial_price=400.0,
        round=1,
        max_rounds=10,
        opponent_reputation=3.0,
        opponent_reputation_summary="",
        metadata={"price_history": [("buyer", 149.0), ("seller", 505.0), ("buyer", 164.0)]},
    )
    decision = StrategyRegistry.instantiate("anchoring").decide(ctx)
    assert decision.action == "counter"
    assert 164.0 < decision.price < 505.0


def _tft_context(history, current_price):
    return NegotiationContext(
        role="buyer",
        max_budget=500.0,
        min_price=240.0,
        current_price=current_price,
        current_sla="Standard",
        initial_price=400.0,
        round=1,
        max_rounds=10,
        opponent_reputation=3.0,
        opponent_reputation_summary="",
        metadata={"price_history": history},
    )


def test_tit_for_tat_does_not_mirror_an_opponent_moving_away():
    # A single seller ask above the listing price is not a concession.
    strategy = StrategyRegistry.instantiate("tit_for_tat", generosity=0.0)
    ctx = _tft_context([("buyer", 370.0), ("seller", 589.0)], 589.0)
    decision = strategy.decide(ctx)
    # Nothing mirrored: only the (tiny, round-1) deadline concession applies.
    assert decision.price == ctx.with_deadline(370.0, strategy.deadline_firmness)
    assert decision.price < 375.0


def test_tit_for_tat_mirrors_the_opponents_absolute_step():
    strategy = StrategyRegistry.instantiate("tit_for_tat", generosity=0.0)
    history = [("buyer", 300.0), ("seller", 480.0), ("buyer", 310.0), ("seller", 460.0)]
    decision = strategy.decide(_tft_context(history, 460.0))
    assert decision.price == 330.0  # its own 310 + the seller's 20-unit step
