"""Tests for the Strategy Framework (base, registry, builtins)."""

from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import StrategyRegistry


def _buyer_context(**overrides) -> NegotiationContext:
    """Create a default buyer NegotiationContext for testing."""
    defaults = dict(
        role="buyer",
        max_budget=500.0,
        min_price=100.0,
        current_price=350.0,
        current_sla="Standard",
        initial_price=400.0,
        round=2,
        max_rounds=10,
        opponent_reputation=3.5,
        opponent_reputation_summary="Average seller.",
        message_history=[],
    )
    defaults.update(overrides)
    return NegotiationContext(**defaults)


def _seller_context(**overrides) -> NegotiationContext:
    """Create a default seller NegotiationContext for testing."""
    defaults = dict(
        role="seller",
        max_budget=500.0,
        min_price=100.0,
        current_price=250.0,
        current_sla="Standard",
        initial_price=400.0,
        round=2,
        max_rounds=10,
        opponent_reputation=4.0,
        opponent_reputation_summary="Good buyer.",
        message_history=[],
    )
    defaults.update(overrides)
    return NegotiationContext(**defaults)


# -- Registry -------------------------------------------------------


def test_registry_lists_all_builtins():
    names = StrategyRegistry.names()
    assert "aggressive" in names
    assert "cooperative" in names
    assert "tit_for_tat" in names
    assert "anchoring" in names
    assert "concession" in names
    assert len(names) >= 5


def test_registry_instantiate():
    s = StrategyRegistry.instantiate("cooperative")
    assert isinstance(s, NegotiationStrategy)
    assert s.name == "cooperative"


def test_registry_list_all_metadata():
    items = StrategyRegistry.list_all()
    assert len(items) >= 5
    for item in items:
        assert "name" in item
        assert "class" in item
        assert "description" in item


# -- NegotiationContext properties ----------------------------------


def test_context_zopa_exists():
    ctx = _buyer_context(max_budget=500, min_price=100)
    assert ctx.zopa_exists is True
    assert ctx.zopa_midpoint == 300.0


def test_context_zopa_not_exists():
    ctx = _buyer_context(max_budget=100, min_price=500)
    assert ctx.zopa_exists is False


def test_context_time_pressure():
    ctx0 = _buyer_context(round=0, max_rounds=10)
    ctx5 = _buyer_context(round=5, max_rounds=10)
    ctx9 = _buyer_context(round=9, max_rounds=10)
    assert ctx0.time_pressure == 0.0
    assert ctx5.time_pressure == 0.5
    assert ctx9.time_pressure == 0.9


def test_context_rounds_remaining():
    ctx = _buyer_context(round=7, max_rounds=10)
    assert ctx.rounds_remaining == 3


def test_context_time_pressure_saturates_past_the_deadline():
    """Overdue rounds must clamp to 1.0 rather than exceed it."""
    assert _buyer_context(round=10, max_rounds=10).time_pressure == 1.0
    assert _buyer_context(round=15, max_rounds=10).time_pressure == 1.0
    # A zero round budget means "decide now".
    assert _buyer_context(round=0, max_rounds=0).time_pressure == 1.0


def test_context_budget_utilization():
    """Public SDK helper advertised in the README — keep it verified."""
    assert _buyer_context(current_price=250.0, max_budget=500.0).budget_utilization == 0.5
    # Clamped: an over-budget offer reads as fully utilizing the budget.
    assert _buyer_context(current_price=900.0, max_budget=500.0).budget_utilization == 1.0
    # A zero budget would otherwise divide by zero.
    assert _buyer_context(current_price=10.0, max_budget=0.0).budget_utilization == 1.0


def test_context_seller_margin():
    """Margin above the floor, never negative."""
    assert _seller_context(current_price=250.0, min_price=100.0).seller_margin == 150.0
    assert _seller_context(current_price=100.0, min_price=100.0).seller_margin == 0.0
    assert _seller_context(current_price=50.0, min_price=100.0).seller_margin == 0.0


def test_context_zopa_midpoint_falls_back_without_a_zopa():
    """With no overlap the "fair" point is undefined; the current price is returned."""
    ctx = _buyer_context(max_budget=100.0, min_price=200.0, current_price=250.0)
    assert ctx.zopa_exists is False
    assert ctx.zopa_midpoint == 250.0


# -- Decision guardrails -------------------------------------------


def test_negative_strategy_price_is_clamped_by_the_guardrails():
    """Decision is a plain value; the guardrails are the one place prices are fixed."""
    from agent_bargain.agents.guardrails import apply_guardrails

    d = Decision(action="counter", price=-50.0, sla="Standard")
    assert d.price == -50.0
    g = apply_guardrails(
        action=d.action,
        price=d.price,
        sla=d.sla,
        reasoning=d.reasoning,
        role="buyer",
        limit=500.0,
        current_price=400.0,
    )
    assert g.price == 0.0
    assert "Guardrail" in g.reasoning


# -- Builtin strategies produce valid decisions ---------------------


def test_aggressive_returns_decision():
    s = StrategyRegistry.instantiate("aggressive")
    d = s.decide(_buyer_context())
    assert d.action in ("accept", "counter", "reject")
    assert d.price >= 0


def test_cooperative_returns_decision():
    s = StrategyRegistry.instantiate("cooperative")
    d = s.decide(_seller_context())
    assert d.action in ("accept", "counter", "reject")
    assert d.price >= 0


def test_tit_for_tat_returns_decision():
    s = StrategyRegistry.instantiate("tit_for_tat")
    d = s.decide(_buyer_context(round=0))
    assert d.action in ("accept", "counter", "reject")


def test_anchoring_returns_decision():
    s = StrategyRegistry.instantiate("anchoring")
    d = s.decide(_buyer_context(round=0))
    assert d.action in ("accept", "counter", "reject")


def test_concession_returns_decision():
    s = StrategyRegistry.instantiate("concession")
    d = s.decide(_seller_context())
    assert d.action in ("accept", "counter", "reject")


def test_aggressive_buyer_round0_starts_low():
    s = StrategyRegistry.instantiate("aggressive")
    ctx = _buyer_context(round=0, current_price=400)
    d = s.decide(ctx)
    assert d.action == "counter"
    assert d.price < ctx.current_price  # Should open lower


def test_cooperative_moves_toward_midpoint():
    s = StrategyRegistry.instantiate("cooperative")
    ctx = _buyer_context(current_price=200)  # Already below midpoint (300)
    d = s.decide(ctx)
    # Cooperative should accept since price < Nash midpoint
    assert d.action == "accept" or (d.action == "counter" and d.price >= ctx.current_price)


def test_concession_rejects_no_zopa():
    s = StrategyRegistry.instantiate("concession")
    ctx = _buyer_context(max_budget=50, min_price=500)  # No ZOPA
    d = s.decide(ctx)
    assert d.action == "reject"


# -- Statelessness --------------------------------------------------


def test_tit_for_tat_has_no_mutable_instance_state():
    s = StrategyRegistry.instantiate("tit_for_tat")
    assert not any(name.startswith("_") for name in vars(s))


def test_anchoring_has_no_mutable_instance_state():
    s = StrategyRegistry.instantiate("anchoring")
    assert not any(name.startswith("_") for name in vars(s))


def test_tit_for_tat_decision_depends_only_on_context():
    """Reusing one instance across negotiations must not leak history."""
    shared = StrategyRegistry.instantiate("tit_for_tat")

    ctx_a = _buyer_context(
        round=1,
        current_price=300.0,
        metadata={
            "price_history": [
                ("buyer", 400.0),
                ("seller", 350.0),
                ("buyer", 320.0),
                ("seller", 300.0),
            ]
        },
    )
    ctx_b = _buyer_context(
        round=1,
        current_price=300.0,
        metadata={
            "price_history": [
                ("buyer", 400.0),
                ("seller", 400.0),
                ("buyer", 320.0),
                ("seller", 300.0),
            ]
        },
    )

    shared.decide(ctx_a)  # pollute any hypothetical internal state
    first_b = shared.decide(ctx_b)
    first_b_again = shared.decide(ctx_b)
    fresh_b = StrategyRegistry.instantiate("tit_for_tat").decide(ctx_b)

    assert first_b.price == first_b_again.price
    assert first_b.price == fresh_b.price


def test_anchoring_repeat_calls_are_idempotent():
    s = StrategyRegistry.instantiate("anchoring")
    ctx = _seller_context(round=2)
    first = s.decide(ctx)
    second = s.decide(ctx)
    assert first.price == second.price
    assert first.action == second.action
