"""
Predefined negotiation scenarios for benchmarking.

Each scenario defines the buyer/seller constraints and an initial price,
representing different market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """A single negotiation scenario configuration."""

    name: str
    description: str
    buyer_max: float
    seller_min: float
    initial_price: float
    max_rounds: int = 10

    @property
    def zopa_size(self) -> float:
        return max(0.0, self.buyer_max - self.seller_min)


# -- Built-in scenarios ---------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="balanced",
        description="Equal bargaining power — symmetric ZOPA",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        max_rounds=10,
    ),
    Scenario(
        name="tight_margin",
        description="Narrow ZOPA — small room for agreement",
        buyer_max=250.0,
        seller_min=200.0,
        initial_price=240.0,
        max_rounds=8,
    ),
    Scenario(
        name="buyer_advantage",
        description="Buyer has much more budget than seller needs",
        buyer_max=1000.0,
        seller_min=100.0,
        initial_price=500.0,
        max_rounds=10,
    ),
    Scenario(
        name="seller_advantage",
        description="Seller's minimum is close to buyer's max",
        buyer_max=300.0,
        seller_min=250.0,
        initial_price=290.0,
        max_rounds=6,
    ),
    Scenario(
        name="high_stakes",
        description="Large absolute values — enterprise deal",
        buyer_max=50000.0,
        seller_min=20000.0,
        initial_price=45000.0,
        max_rounds=12,
    ),
    Scenario(
        name="no_zopa",
        description="Impossible deal — no overlap in ranges",
        buyer_max=100.0,
        seller_min=200.0,
        initial_price=250.0,
        max_rounds=5,
    ),
    Scenario(
        name="quick_deal",
        description="Very few rounds — pressure to agree fast",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=350.0,
        max_rounds=3,
    ),
]


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name."""
    for s in SCENARIOS:
        if s.name == name:
            return s
    available = ", ".join(s.name for s in SCENARIOS)
    raise KeyError(f"Scenario {name!r} not found. Available: [{available}]")
