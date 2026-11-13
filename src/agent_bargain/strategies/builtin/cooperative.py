"""
Cooperative Strategy — prioritizes reaching a deal over maximizing individual gain.

Moves toward the ZOPA midpoint (Nash bargaining solution) at a steady pace,
rewarding high-reputation opponents with faster convergence.
"""

from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import strategy


@strategy("cooperative")
class CooperativeStrategy(NegotiationStrategy):
    """Moves steadily toward the ZOPA midpoint (Nash solution). Rewards good reputation."""

    description = (
        "Prioritizes reaching a deal. Moves toward the Nash bargaining midpoint at a "
        "steady pace and accepts slightly worse terms from high-reputation opponents."
    )

    def __init__(self, convergence_rate: float = 0.25) -> None:
        """
        Args:
            convergence_rate: Fraction of the gap to close each round (0.1–0.5).
        """
        self.convergence_rate = max(0.05, min(0.5, convergence_rate))

    #: Linear deadline curve: concedes steadily rather than late.
    deadline_firmness = 1.0

    def decide(self, context: NegotiationContext) -> Decision:
        if not context.zopa_exists:
            return Decision(
                action="reject",
                price=context.current_price,
                sla=context.current_sla,
                reasoning="No ZOPA — deal is impossible, rejecting gracefully.",
            )

        target = context.zopa_midpoint

        # Reputation bonus: high-rep opponent gets slightly more favourable target
        rep_bonus = (context.opponent_reputation / 5.0) * 0.05
        if context.role == "buyer":
            # Willing to pay slightly more to a trusted seller
            target = min(context.max_budget, target * (1 + rep_bonus))
        else:
            # Willing to accept slightly less from a trusted buyer
            target = max(context.min_price, target * (1 - rep_bonus))

        # Accept if current price is within 2% of our target or better
        acceptance_threshold = abs(target) * 0.02
        if context.role == "buyer" and context.current_price <= target + acceptance_threshold:
            return Decision(
                action="accept",
                price=context.current_price,
                sla=context.current_sla,
                reasoning=f"Price ${context.current_price:.2f} ≤ cooperative target ${target:.2f}.",
            )
        if context.role == "seller" and context.current_price >= target - acceptance_threshold:
            return Decision(
                action="accept",
                price=context.current_price,
                sla=context.current_sla,
                reasoning=f"Price ${context.current_price:.2f} ≥ cooperative target ${target:.2f}.",
            )

        # Close a fixed fraction of the gap between our own last offer and the
        # target (never backwards, should the estimated target move behind us).
        gap = target - context.own_position
        if context.role == "seller":
            gap = -gap
        # The target is a midpoint of an *estimated* ZOPA, so it can sit on
        # the wrong side of the opponent's real limit; the linear deadline
        # curve keeps conceding past it rather than stalling until timeout.
        new_price = context.with_deadline(
            context.concede(gap * self.convergence_rate), self.deadline_firmness
        )
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
                f"Cooperative move toward Nash midpoint ${target:.2f} "
                f"(convergence_rate={self.convergence_rate:.2f}, rep_bonus={rep_bonus:.3f})."
            ),
        )
