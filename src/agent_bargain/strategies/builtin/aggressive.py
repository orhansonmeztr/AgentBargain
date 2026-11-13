"""
Aggressive Strategy — maximizes individual gain with minimal concession.

The aggressive negotiator starts far from their real constraint and concedes
as little as possible, then concedes steeply near the deadline (a Boulware
curve) so a deal that is possible is still reached.
"""

from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import strategy


@strategy("aggressive")
class AggressiveStrategy(NegotiationStrategy):
    """Tough negotiator: starts extreme, concedes slowly, only softens near timeout."""

    description = (
        "Maximizes individual gain. Starts with an extreme opening offer and concedes "
        "very slowly, softening only under strong time pressure near the deadline."
    )

    def __init__(self, aggressiveness: float = 0.8) -> None:
        """
        Args:
            aggressiveness: 0.0 (gentle) → 1.0 (maximally aggressive). Default 0.8.
        """
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))

    @property
    def deadline_firmness(self) -> float:
        """Boulware exponent of the deadline curve: the tougher, the later it concedes."""
        return 1.0 + 4.0 * self.aggressiveness

    def decide(self, context: NegotiationContext) -> Decision:
        if not context.zopa_exists:
            return Decision(
                action="reject",
                price=context.current_price,
                sla=context.current_sla,
                reasoning="No ZOPA detected. Rejecting.",
            )

        if context.round == 0:
            price = self._opening_price(context)
            reasoning = f"Aggressive opening offer as {context.role}."
        else:
            price = self._counter_price(context)
            reasoning = (
                f"Aggressive counter (aggressiveness={self.aggressiveness:.1f}, "
                f"time_pressure={context.time_pressure:.2f})."
            )
        # Hold firm, then concede to the own limit by the last round instead
        # of timing out with a deal still possible.
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

    def _opening_price(self, context: NegotiationContext) -> float:
        if context.role == "buyer":
            # Start very low — far below budget
            price = (
                context.min_price
                + (context.max_budget - context.min_price) * (1 - self.aggressiveness) * 0.3
            )
            price = max(context.min_price * 0.5, price)
        else:
            # Start very high — far above minimum
            price = context.min_price + (context.max_budget - context.min_price) * (
                0.7 + self.aggressiveness * 0.3
            )
        return round(price, 2)

    def _counter_price(self, context: NegotiationContext) -> float:
        """Concede a small amount from our own last offer, growing with time pressure."""
        concession_rate = (1 - self.aggressiveness) * 0.05 + context.time_pressure * 0.1
        zopa_size = context.max_budget - context.min_price
        return context.concede(zopa_size * concession_rate)
