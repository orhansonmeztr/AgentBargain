"""
Tit-for-Tat Strategy — classic game theory reciprocity.

Mirrors the opponent's last move: if they were cooperative, be cooperative;
if they were aggressive (small concession), be aggressive back.
Starts cooperative to signal good faith.

Stateless by design: inter-round prices come from
``NegotiationContext.metadata["price_history"]`` rather than instance state.
"""

from agent_bargain.strategies.base import Decision, NegotiationContext, NegotiationStrategy
from agent_bargain.strategies.registry import strategy


def _opponent_concession(context: NegotiationContext) -> float:
    """How far (in price) the opponent moved toward us between its last two offers.

    Directional and measured between the opponent's *own* offers: a seller
    lowering its ask (or a buyer raising its bid) is a concession; moving away
    is not, and counts as zero. Measuring ``abs()`` against the listing price
    instead read an anchoring seller *raising* its ask from 400 to 589 as a
    47% concession, and tit-for-tat "mirrored" it by jumping to its budget.
    The step is mirrored in absolute terms: as a percentage of the opponent's
    price applied to the ZOPA, a 20-unit step on a 149 bid became a 70-unit
    concession.
    """
    opponent = "seller" if context.role == "buyer" else "buyer"
    offers = [price for actor, price in context.price_history if actor == opponent]
    if len(offers) < 2:
        return 0.0  # nothing to mirror yet
    previous, latest = offers[-2], offers[-1]
    moved = previous - latest if context.role == "buyer" else latest - previous
    return max(0.0, moved)


@strategy("tit_for_tat")
class TitForTatStrategy(NegotiationStrategy):
    """Classic game theory reciprocity: start cooperative, mirror opponent's moves."""

    description = (
        "Mirrors the opponent's behavior: cooperative when they cooperate, "
        "firm when they are firm. Starts with a cooperative opening to signal good faith."
    )

    def __init__(self, generosity: float = 0.1) -> None:
        """
        Args:
            generosity: Extra concession on top of the mirrored step, as a
                        fraction of a quarter of the estimated ZOPA (0–0.3).
                        Positive generosity prevents lock-ins in mutual aggression.
        """
        self.generosity = max(0.0, min(0.3, generosity))

    #: Moderate Boulware curve for the deadline fallback.
    deadline_firmness = 2.0

    def decide(self, context: NegotiationContext) -> Decision:
        # First round: cooperative opening
        if context.round == 0:
            return self._cooperative_open(context)

        opponent_concession = _opponent_concession(context)

        # Mirror the opponent's step, plus a generosity bonus so two mirrors
        # facing each other do not lock in place.
        zopa_size = max(1.0, context.max_budget - context.min_price)
        concession = opponent_concession + zopa_size * self.generosity * 0.25
        # Mirroring a stubborn opponent can lock both in place; the deadline
        # curve still brings this side to its own limit by the last round.
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
                f"Tit-for-tat: mirroring opponent concession ${opponent_concession:.2f} "
                f"+ generosity {self.generosity:.2%}."
            ),
        )

    def _cooperative_open(self, context: NegotiationContext) -> Decision:
        """Open near the midpoint of ZOPA to signal cooperative intent."""
        if context.zopa_exists:
            price = context.zopa_midpoint
        elif context.role == "buyer":
            price = context.max_budget * 0.75
        else:
            price = context.min_price * 1.25
        return Decision(
            action="counter",
            price=round(price, 2),
            sla=context.current_sla,
            reasoning="Tit-for-tat: cooperative opening offer near ZOPA midpoint.",
        )
