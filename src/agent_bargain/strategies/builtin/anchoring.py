"""
Anchoring Strategy — exploit first-mover advantage via extreme opening offer.

Classic behavioral economics: the first number heard in a negotiation
disproportionately influences the final outcome. This strategy opens with
an extreme anchor, then makes small concessions to appear reasonable.
"""

from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import strategy


@strategy("anchoring")
class AnchoringStrategy(NegotiationStrategy):
    """Exploit first-mover advantage: anchor extreme, then concede slowly."""

    description = (
        "Opens with an extreme anchor to set the reference point. "
        "Makes small concessions each round to appear reasonable while protecting gains."
    )

    def __init__(
        self,
        anchor_strength: float = 0.7,
        concession_decay: float = 0.6,
    ) -> None:
        """
        Args:
            anchor_strength: How extreme the opening anchor is (0=mild, 1=extreme).
            concession_decay: Each round, concession size multiplies by this factor
                              (0.5=halving, 1.0=constant).
        """
        self.anchor_strength = max(0.0, min(1.0, anchor_strength))
        self.concession_decay = max(0.3, min(1.0, concession_decay))

    def decide(self, context: NegotiationContext) -> Decision:
        if not context.zopa_exists:
            return Decision(
                action="reject",
                price=context.current_price,
                sla=context.current_sla,
                reasoning="No ZOPA — rejecting.",
            )

        # Round 0: set the extreme anchor; afterwards, decaying concessions.
        if context.round == 0:
            price = round(self._compute_anchor(context), 2)
            reasoning = (
                f"Anchoring: extreme opening offer ${price:.2f} (strength={self.anchor_strength})."
            )
        else:
            # Concession shrinks each round (anchoring locks in early advantage).
            # Derived from the context every call — no instance state.
            concession = self._base_concession(context) * (self.concession_decay**context.round)
            concession = max(0.01, concession)  # never zero
            price = context.concede(concession)
            reasoning = (
                f"Anchoring concession ${concession:.2f} "
                f"(round {context.round}, decay={self.concession_decay})."
            )
        # The decay alone would stall short of a deal; the deadline curve
        # still walks the anchor to the own limit by the last round.
        price = context.with_deadline(price, self.deadline_firmness)

        if context.offer_is_no_worse_than(price):
            return Decision(
                action="accept",
                price=context.current_price,
                sla=context.current_sla,
                reasoning=(
                    f"Offer ${context.current_price:.2f} is no worse than my next "
                    f"counter ${price:.2f} — accepting."
                ),
            )

        return Decision(action="counter", price=price, sla=context.current_sla, reasoning=reasoning)

    @property
    def deadline_firmness(self) -> float:
        """Boulware exponent of the deadline curve: a stronger anchor holds longer."""
        return 1.0 + 3.0 * self.anchor_strength

    def _base_concession(self, context: NegotiationContext) -> float:
        """Baseline per-round concession before exponential decay."""
        zopa_size = max(0.0, context.max_budget - context.min_price)
        return zopa_size * 0.15 * (1 - self.anchor_strength * 0.5)

    def _compute_anchor(self, context: NegotiationContext) -> float:
        zopa_size = context.max_budget - context.min_price
        overshoot = zopa_size * self.anchor_strength

        if context.role == "buyer":
            # Anchor low — far below min_price even
            return max(0.01, context.min_price - overshoot * 0.5)
        else:
            # Anchor high — far above max_budget
            return context.max_budget + overshoot * 0.5
