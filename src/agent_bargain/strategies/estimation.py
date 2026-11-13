"""Estimating the opponent's private limit from public information.

A real negotiator never sees the other side's walk-away point: the buyer does
not know the seller's floor, the seller does not know the buyer's budget. The
strategy layer used to receive both true values, so every built-in strategy
computed its "ZOPA" and "Nash midpoint" from a secret it could never have —
and the benchmark ranked omniscient agents that cannot exist in the product.

Strategies now get their *own* limit exactly and the opponent's as an
estimate built only from what both parties can observe:

* ``initial_price`` — the listing's asking price (public);
* the opponent's offers so far, from ``price_history``.

The estimate is deliberately simple and deterministic, so the strategy graph
and the benchmark produce identical trajectories
(``tests/test_benchmark_fidelity.py``):

* **Buyer → seller floor.** A seller never asks below its floor, so its lowest
  ask (or the opening price) is an upper bound; the floor is assumed to sit
  :data:`ASSUMED_SELLER_MARGIN` below it.
* **Seller → buyer budget.** A buyer never bids above its budget, so its highest
  bid is a lower bound; the budget is assumed to sit
  :data:`ASSUMED_BUYER_HEADROOM` above it. A buyer who came to a listing at
  ``initial_price`` is assumed able to afford roughly that, so the estimate
  never drops below it — otherwise a single lowball bid would make the seller
  believe no deal is possible and walk away.

Guardrails still enforce each party's *true* limit, so a wrong estimate can
cost a party surplus or a deal, never a breach of its own constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

Role = Literal["buyer", "seller"]

#: Fraction of the seller's lowest observed ask assumed to be margin above its floor.
ASSUMED_SELLER_MARGIN = 0.4
#: Fraction above the buyer's highest observed bid assumed to be unspent budget.
ASSUMED_BUYER_HEADROOM = 0.25


def estimate_opponent_limit(
    role: Role,
    *,
    initial_price: float,
    price_history: Sequence[tuple[str, float]],
) -> float:
    """Estimate the opponent's private limit as seen by ``role``.

    Returns the estimated seller floor for a buyer, or the estimated buyer
    budget for a seller. Uses only public information.
    """
    if role == "buyer":
        asks = [price for actor, price in price_history if actor == "seller"]
        lowest_ask = min([*asks, initial_price])
        return round(max(0.0, lowest_ask * (1 - ASSUMED_SELLER_MARGIN)), 2)

    bids = [price for actor, price in price_history if actor == "buyer"]
    highest_bid = max(bids, default=0.0)
    return round(max(initial_price, highest_bid * (1 + ASSUMED_BUYER_HEADROOM)), 2)


def limits_for(
    role: Role,
    *,
    own_limit: float,
    initial_price: float,
    price_history: Sequence[tuple[str, float]],
) -> tuple[float, float]:
    """Return ``(max_budget, min_price)`` as ``role`` may know them.

    The role's own field is its true limit; the opponent's is an estimate.
    """
    estimate = estimate_opponent_limit(
        role, initial_price=initial_price, price_history=price_history
    )
    if role == "buyer":
        return own_limit, estimate
    return estimate, own_limit
