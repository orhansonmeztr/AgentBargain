"""
Base classes for the AgentBargain Strategy Framework.

Defines the abstract interface that all negotiation strategies must implement,
plus the immutable context and decision data structures passed between layers.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class NegotiationContext:
    """Immutable snapshot of the negotiation state passed to every strategy.

    All information the strategy needs to make a decision is here —
    no global state, no side effects.
    """

    # Role & constraints. The deciding role's own field is its true limit; the
    # opponent's is an *estimate* from public information (listing price and
    # offers so far — see strategies/estimation.py). A strategy must never be
    # handed the other side's real walk-away point.
    role: Literal["buyer", "seller"]
    max_budget: float  # Buyer's budget: true for a buyer, estimated for a seller
    min_price: float  # Seller's floor: true for a seller, estimated for a buyer

    # Current state
    current_price: float
    current_sla: str
    initial_price: float

    # Round tracking
    round: int
    max_rounds: int

    # Opponent context (what the other party said last)
    opponent_reputation: float
    opponent_reputation_summary: str

    # Full message history as plain strings (for advanced strategies)
    message_history: list[str] = field(default_factory=list)

    # Arbitrary extra metadata (for custom strategies)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def own_limit(self) -> float:
        """The deciding role's true limit (budget for a buyer, floor for a seller)."""
        return self.max_budget if self.role == "buyer" else self.min_price

    @property
    def opponent_limit_estimate(self) -> float:
        """The opponent's limit as this role estimates it — never the true value."""
        return self.min_price if self.role == "buyer" else self.max_budget

    @property
    def price_history(self) -> list[tuple[str, float]]:
        """Role-tagged ``(role, price)`` offers so far, oldest first.

        Read from ``metadata["price_history"]``, which the strategy node and
        the benchmark runner both fill; malformed entries are skipped.
        """
        raw = self.metadata.get("price_history") or []
        history: list[tuple[str, float]] = []
        for entry in raw:
            if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
                continue
            try:
                price = float(entry[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(price):
                history.append((str(entry[0]), price))
        return history

    @property
    def own_last_offer(self) -> float | None:
        """This role's most recent offer, or ``None`` before its first turn."""
        for actor, price in reversed(self.price_history):
            if actor == self.role:
                return price
        return None

    @property
    def own_position(self) -> float:
        """Where this role's next concession starts from.

        ``current_price`` is the *opponent's* latest offer, not this role's
        own position: conceding from it made a buyer bid above the seller's
        ask (and a seller ask below the buyer's bid). A role concedes from its
        own last offer; before its first one, a buyer starts from its estimate
        of the seller's floor and a seller from the listing price — its own
        public ask.
        """
        last = self.own_last_offer
        if last is not None:
            return last
        return self.min_price if self.role == "buyer" else self.initial_price

    def concede(self, amount: float) -> float:
        """Price after conceding ``amount`` from :attr:`own_position`.

        A buyer moves up, a seller down, never past its own limit and never
        backwards (a negative ``amount`` concedes nothing). If the result meets
        or crosses ``current_price``, the turn guardrail turns the counter into
        an accept of the price on the table.
        """
        amount = max(0.0, amount)
        if self.role == "buyer":
            return round(min(self.max_budget, self.own_position + amount), 2)
        return round(max(self.min_price, self.own_position - amount), 2)

    @property
    def own_first_offer(self) -> float | None:
        """This role's opening offer, or ``None`` before its first turn."""
        for actor, price in self.price_history:
            if actor == self.role:
                return price
        return None

    @property
    def deadline_progress(self) -> float:
        """0.0 on the first round, 1.0 on the last one this role gets to move in.

        Unlike :attr:`time_pressure` (``round / max_rounds``), this reaches 1.0
        *on* the final round, while there is still a turn left to act on it.
        """
        if self.max_rounds <= 1:
            return 1.0
        return min(1.0, self.round / (self.max_rounds - 1))

    def deadline_price(self, firmness: float) -> float | None:
        """The least-conceded price this role should still be holding out for.

        A time-dependent concession curve from the role's opening offer to its
        own limit, reaching the limit on the last round::

            opening + (own_limit - opening) * deadline_progress ** firmness

        ``firmness`` > 1 holds firm and concedes late (Boulware), < 1 concedes
        early (Conceder), 1 is linear. Because both parties reach their own
        limits by the final round, any negotiation with a real ZOPA ends in a
        deal instead of timing out with the parties still apart.

        Returns ``None`` on the opening round (there is no opening offer to
        start the curve from yet), unless it is also the last round.
        """
        if self.round == 0 and self.deadline_progress < 1.0:
            return None
        start = self.own_first_offer
        if start is None:
            start = self.own_position
        progress = self.deadline_progress ** max(firmness, 0.01)
        return round(start + (self.own_limit - start) * progress, 2)

    def with_deadline(self, price: float, firmness: float) -> float:
        """``price``, conceded further if it lags behind :meth:`deadline_price`."""
        bound = self.deadline_price(firmness)
        if bound is None:
            return price
        if self.role == "buyer":
            return round(min(self.max_budget, max(price, bound)), 2)
        return round(max(self.min_price, min(price, bound)), 2)

    def offer_is_no_worse_than(self, price: float) -> bool:
        """True when the price on the table is at least as good as ``price``.

        The acceptance rule every built-in strategy uses: take the opponent's
        offer once it is no worse than the counter this role would make next
        (``AC_next`` in the automated-negotiation literature). Accepting
        anything inside a fixed band of the role's own limit gave the other
        side most of the surplus — an anchoring seller that opened at 505
        accepted 164 one round later because it was above 1.2× its floor.
        """
        if self.role == "buyer":
            return self.current_price <= price
        return self.current_price >= price

    @property
    def rounds_remaining(self) -> int:
        """How many rounds are left before timeout."""
        return max(0, self.max_rounds - self.round)

    @property
    def time_pressure(self) -> float:
        """Normalized time pressure: 0.0 (no pressure) → 1.0 (last round)."""
        if self.max_rounds == 0:
            return 1.0
        return min(1.0, self.round / self.max_rounds)

    @property
    def zopa_exists(self) -> bool:
        """True if the *estimated* Zone of Possible Agreement is non-empty.

        One bound is the opponent's estimated limit, so this is the decider's
        belief that a deal is possible, not a fact."""
        return self.max_budget >= self.min_price

    @property
    def zopa_midpoint(self) -> float:
        """The Nash bargaining midpoint of the estimated ZOPA."""
        if not self.zopa_exists:
            return self.current_price
        return (self.max_budget + self.min_price) / 2.0

    @property
    def budget_utilization(self) -> float:
        """For buyers: how much of the budget the current price uses (0–1)."""
        if self.max_budget == 0:
            return 1.0
        return min(1.0, self.current_price / self.max_budget)

    @property
    def seller_margin(self) -> float:
        """For sellers: how much above minimum the current price is."""
        return max(0.0, self.current_price - self.min_price)


@dataclass
class Decision:
    """The output of a strategy's decide() call.

    A plain value. Prices are *not* clamped here: every decision passes
    through ``agents.guardrails.apply_guardrails`` before it is acted on, and
    that is the single place the rules (non-finite and negative prices,
    limits, crossing offers) live. A second copy here only ever drifted.
    """

    action: Literal["accept", "counter", "reject"]
    price: float
    sla: str = "Standard"
    reasoning: str = ""


class NegotiationStrategy(ABC):
    """Abstract base class for all negotiation strategies.

    Subclass this to create a custom strategy, then register it with
    the @strategy decorator or StrategyRegistry.register().

    Example::

        @strategy("tit_for_tat")
        class TitForTatStrategy(NegotiationStrategy):
            def decide(self, context: NegotiationContext) -> Decision:
                # Mirror what the opponent did last round
                ...
    """

    #: Human-readable name shown in benchmark reports and the dashboard.
    name: str = "unnamed"

    #: Short description shown in the API and dashboard UI.
    description: str = ""

    @abstractmethod
    def decide(self, context: NegotiationContext) -> Decision:
        """Given the current negotiation context, return a decision.

        Args:
            context: Immutable snapshot of the current negotiation state.

        Returns:
            A Decision with action, proposed price, SLA, and reasoning.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
