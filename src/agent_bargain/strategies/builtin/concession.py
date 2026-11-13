"""
Time-Based Concession Strategy — concedes more as the deadline approaches.

Models the classical deadline effect: negotiators make larger concessions
as time runs out. The concession curve can be tuned from linear to convex.
"""

from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import strategy


@strategy("concession")
class ConcessionStrategy(NegotiationStrategy):
    """Deadline-driven: concessions accelerate as rounds run out (classical deadline effect)."""

    description = (
        "Concession rate increases as the deadline approaches — models the classical "
        "'deadline effect'. Configurable curve: linear, convex, or concave."
    )

    def __init__(
        self,
        curve: str = "convex",
        max_concession_pct: float = 0.4,
    ) -> None:
        """
        Args:
            curve: Concession curve shape. One of 'linear', 'convex', 'concave'.
                   - 'linear':  constant rate throughout
                   - 'convex':  slow at first, fast at end (classical deadline effect)
                   - 'concave': fast at first, slow at end (front-loaded)
            max_concession_pct: Maximum total concession as fraction of ZOPA (0–1).
        """
        if curve not in ("linear", "convex", "concave"):
            raise ValueError(f"curve must be 'linear', 'convex', or 'concave', got {curve!r}")
        self.curve = curve
        self.max_concession_pct = max(0.05, min(1.0, max_concession_pct))

    def decide(self, context: NegotiationContext) -> Decision:
        if not context.zopa_exists:
            return Decision(
                action="reject",
                price=context.current_price,
                sla=context.current_sla,
                reasoning="No ZOPA — impossible deal, rejecting.",
            )

        concession = self._concession_amount(context)
        new_price = context.with_deadline(context.concede(concession), self.deadline_firmness)

        if context.offer_is_no_worse_than(new_price):
            return Decision(
                action="accept",
                price=context.current_price,
                sla=context.current_sla,
                reasoning=(
                    f"Offer ${context.current_price:.2f} is no worse than my next "
                    f"counter ${new_price:.2f} — accepting."
                ),
            )

        return Decision(
            action="counter",
            price=new_price,
            sla=context.current_sla,
            reasoning=(
                f"Concession strategy ({self.curve} curve), "
                f"time_pressure={context.time_pressure:.2f}, concession=${concession:.2f}."
            ),
        )

    @property
    def deadline_firmness(self) -> float:
        """Exponent of the deadline curve, matching the configured curve shape."""
        return {"linear": 1.0, "convex": 2.0, "concave": 0.5}[self.curve]

    def _concession_amount(self, context: NegotiationContext) -> float:
        """This round's step along the configured total-concession curve.

        The curve gives the *cumulative* share of ``max_concession`` conceded
        by time ``t``; a round concedes the difference since the previous one.
        """
        zopa_size = context.max_budget - context.min_price
        max_concession = zopa_size * self.max_concession_pct
        t = context.time_pressure  # 0.0 → 1.0

        if context.round > 0 and context.max_rounds > 1:
            prev_t = (context.round - 1) / context.max_rounds
            step = self._curve_value(t) - self._curve_value(prev_t)
            return max(0.01, max_concession * step)

        return max_concession * self._curve_value(t) * 0.15  # opening round concession

    def _curve_value(self, t: float) -> float:
        """Cumulative concession share at time ``t`` for the configured curve."""
        if self.curve == "linear":
            return t
        if self.curve == "convex":
            # Slow at first, fast near the deadline (classical deadline effect).
            return t**2
        # Concave: fast at first, slow near the deadline.
        return 1 - (1 - t) ** 2
