"""Turn guardrails.

* A counter must stay within the offering party's own limit. Only the
  *accepting* side used to be checked, so a buyer could offer above its budget
  (or a seller below its floor) and the counterparty's accept would close the
  deal outside the ZOPA.
* A counter that meets or crosses the price on the table is an accept. A buyer
  used to bid *above* the seller's ask (476 asked, 484 offered) and the price
  ratcheted past both openings instead of converging between them.
"""

import pytest

from agent_bargain.agents.guardrails import apply_guardrails


def _guard(**overrides):
    kwargs = {
        "action": "counter",
        "price": 300.0,
        "sla": "Standard",
        "reasoning": "",
        "role": "buyer",
        "limit": 500.0,
        "current_price": 400.0,
    }
    kwargs.update(overrides)
    return apply_guardrails(**kwargs)


def test_buyer_counter_above_budget_is_capped():
    g = _guard(price=900.0, current_price=950.0, limit=500.0)
    assert g.action == "counter"
    assert g.price == 500.0
    assert "capped at own budget" in g.reasoning
    assert "500" not in g.reasoning, "the note must not reveal the limit"


def test_seller_counter_below_floor_is_raised():
    g = _guard(role="seller", price=50.0, current_price=30.0, limit=100.0)
    assert g.action == "counter"
    assert g.price == 100.0
    assert "raised to own minimum" in g.reasoning


def test_in_limit_counter_is_untouched():
    g = _guard(price=350.0)
    assert (g.action, g.price, g.reasoning) == ("counter", 350.0, "")


def test_buyer_bid_above_the_ask_accepts_the_ask():
    # The trace that found this: seller asks 476, buyer "counters" 484.
    g = _guard(price=484.0, current_price=476.0, limit=500.0)
    assert (g.action, g.price) == ("accept", 476.0)
    assert "meets the price on the table" in g.reasoning


def test_seller_ask_below_the_bid_accepts_the_bid():
    g = _guard(role="seller", price=180.0, current_price=200.0, limit=100.0)
    assert (g.action, g.price) == ("accept", 200.0)


def test_unchanged_counter_settles_instead_of_spinning():
    for role, limit in (("buyer", 500.0), ("seller", 100.0)):
        g = _guard(role=role, price=300.0, current_price=300.0, limit=limit)
        assert (g.action, g.price) == ("accept", 300.0)


def test_capped_counter_that_reaches_the_ask_accepts_it():
    # Capped at the 500 budget, which is still at least the 480 ask.
    g = _guard(price=600.0, current_price=480.0, limit=500.0)
    assert (g.action, g.price) == ("accept", 480.0)


def test_capped_counter_short_of_the_ask_stays_a_counter():
    g = _guard(price=600.0, current_price=550.0, limit=500.0)
    assert (g.action, g.price) == ("counter", 500.0)


def test_accept_outside_own_limit_is_still_rejected():
    g = _guard(action="accept", price=0.0, current_price=600.0, limit=500.0)
    assert g.action == "reject"


# -- accept that names a better price is a counter ---------------------------


def test_buyer_accept_naming_a_lower_price_becomes_a_counter():
    """The demo case: "a price closer to $120" + accept, with $130 on the table."""
    g = _guard(action="accept", price=120.0, current_price=130.0, limit=150.0)
    assert (g.action, g.price) == ("counter", 120.0)
    assert "treated as a counter" in g.reasoning


def test_seller_accept_naming_a_higher_price_becomes_a_counter():
    g = _guard(role="seller", action="accept", price=140.0, current_price=120.0, limit=50.0)
    assert (g.action, g.price) == ("counter", 140.0)


def test_accept_to_counter_still_respects_the_own_limit():
    # The seller's "better" price is below its floor: raised to the floor, and
    # since that still beats the buyer's bid it stays a counter.
    g = _guard(role="seller", action="accept", price=95.0, current_price=80.0, limit=100.0)
    assert (g.action, g.price) == ("counter", 100.0)


@pytest.mark.parametrize("price", [0.0, -5.0, 129.5, 130.0, 135.0])
def test_accept_is_kept_when_the_price_is_not_a_clear_better_offer(price):
    """Zero/negative is hallucination, within 1% is rounding, worse is just echo."""
    g = _guard(action="accept", price=price, current_price=130.0, limit=150.0)
    assert (g.action, g.price) == ("accept", 130.0)


def test_accept_with_nan_price_still_settles_at_the_table():
    import math

    g = _guard(action="accept", price=math.nan, current_price=130.0, limit=150.0)
    assert (g.action, g.price) == ("accept", 130.0)


def test_turn_text_does_not_double_the_sla_period():
    from agent_bargain.agents.guardrails import GuardedDecision, build_turn_message

    msg = build_turn_message(
        role="buyer", decision=GuardedDecision("accept", 650.0, "Standard SLA with testing.", "ok")
    )
    assert msg.additional_kwargs["public_content"] == (
        "BUYER: ACCEPT - Price: $650.0, SLA: Standard SLA with testing."
    )
    assert ".." not in str(msg.content)


# -- no retraction ------------------------------------------------------------


def test_seller_cannot_raise_its_own_previous_ask():
    g = _guard(role="seller", price=670.0, current_price=630.0, limit=200.0, own_last_offer=650.0)
    assert (g.action, g.price) == ("counter", 650.0)
    assert "own previous ask" in g.reasoning


def test_buyer_cannot_lower_its_own_previous_bid():
    g = _guard(price=580.0, current_price=650.0, limit=800.0, own_last_offer=600.0)
    assert (g.action, g.price) == ("counter", 600.0)
    assert "own previous bid" in g.reasoning


def test_concessions_are_untouched():
    assert (
        _guard(price=630.0, current_price=650.0, limit=800.0, own_last_offer=600.0).price == 630.0
    )
    seller = _guard(
        role="seller", price=640.0, current_price=630.0, limit=200.0, own_last_offer=650.0
    )
    assert seller.price == 640.0


def test_first_turn_may_anchor_above_the_listing():
    """The listing price is not an offer: an opening ask above it is allowed."""
    g = _guard(role="seller", price=700.0, current_price=600.0, limit=200.0, own_last_offer=None)
    assert (g.action, g.price) == ("counter", 700.0)


def test_the_demo_ratchet_now_settles():
    """Replay of the timed-out demo: the seller's ask rose 650 -> 670 -> 680
    while the buyer chased it. Held at 650, the buyer's 660 crosses and settles."""
    last = {"buyer": None, "seller": None}
    table = 650.0  # listing
    outcome = None
    for role, price in [
        ("buyer", 600),
        ("seller", 650),
        ("buyer", 630),
        ("seller", 670),
        ("buyer", 660),
    ]:
        limit = 800.0 if role == "buyer" else 200.0
        g = _guard(
            role=role,
            price=float(price),
            current_price=table,
            limit=limit,
            own_last_offer=last[role],
        )
        last[role], table = g.price, g.price
        if g.action == "accept":
            outcome = (role, g.price)
            break
    assert outcome == ("buyer", 650.0)
