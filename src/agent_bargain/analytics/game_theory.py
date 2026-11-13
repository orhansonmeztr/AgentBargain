"""
Game Theory Analytics for AgentBargain.

Computes Nash Bargaining Solution, Pareto efficiency, ZOPA analysis,
BATNA distances, and concession curve metrics for completed negotiations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ZOPAAnalysis:
    """Zone of Possible Agreement analysis."""

    lower_bound: float  # Seller's min price (bottom of ZOPA)
    upper_bound: float  # Buyer's max budget (top of ZOPA)
    size: float  # upper - lower (negative means no ZOPA)
    exists: bool  # True if size >= 0
    utilization: float  # How much of the ZOPA the final price sits in (0–1)


@dataclass
class NashAnalysis:
    """Nash Bargaining Solution analysis."""

    nash_price: float  # The symmetric Nash solution price (ZOPA midpoint)
    actual_price: float  # The agreed price
    distance: float  # |actual - nash| (how far from optimal fairness)
    distance_pct: float  # distance / zopa_size (normalized)
    buyer_favored: bool  # True if deal was better for buyer than Nash


@dataclass
class ParetoAnalysis:
    """Pareto efficiency analysis."""

    is_efficient: bool  # True if deal is within ZOPA (Pareto optimal)
    total_surplus: float  # ZOPA size (max possible joint gain)
    buyer_surplus: float  # max_budget - final_price (buyer's gain)
    seller_surplus: float  # final_price - min_price (seller's gain)
    surplus_ratio: (
        float | None
    )  # buyer_surplus / seller_surplus (fairness); None when seller_surplus == 0
    egalitarian_score: float  # 1 - |buyer_surplus - seller_surplus| / total_surplus
    # 1.0 = perfectly equal split, 0.0 = all to one party


@dataclass
class BATNAAnalysis:
    """Best Alternative to Negotiated Agreement analysis."""

    buyer_batna_distance: float  # max_budget - final_price (how far from walk-away)
    seller_batna_distance: float  # final_price - min_price (how far from walk-away)
    both_above_batna: bool  # True if both parties beat their BATNA


@dataclass
class ConcessionAnalysis:
    """Analysis of the concession pattern during negotiation."""

    total_rounds: int
    buyer_total_concession: float  # Total $ moved by buyer
    seller_total_concession: float  # Total $ moved by seller
    buyer_concession_rate: float  # Avg $ per round
    seller_concession_rate: float  # Avg $ per round
    first_offer: float  # Opening price
    final_price: float  # Agreed price
    price_trajectory: list[float]  # Price at each round
    # Populated only when the trajectory carries actor labels. A "retraction"
    # is a move *against* the party's own interest (hardening back).
    buyer_retraction: float = 0.0
    seller_retraction: float = 0.0
    attributed: bool = False  # True when per-actor attribution was possible


@dataclass
class NegotiationGameTheoryReport:
    """Complete game-theoretic analysis of a single negotiation."""

    negotiation_id: str
    outcome: str  # 'accepted', 'rejected', 'timeout'
    final_price: float | None

    zopa: ZOPAAnalysis
    nash: NashAnalysis
    pareto: ParetoAnalysis
    batna: BATNAAnalysis
    concession: ConcessionAnalysis

    # Overall efficiency score (0–1)
    efficiency_score: float
    fairness_score: float

    # Human-readable summary
    summary: str = field(default="")

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = self._generate_summary()

    def _generate_summary(self) -> str:
        lines = [f"Negotiation {self.negotiation_id} — {self.outcome.upper()}"]
        if self.outcome == "accepted" and self.final_price is not None:
            lines.append(f"  Final price: ${self.final_price:.2f}")
            lines.append(
                f"  ZOPA: ${self.zopa.lower_bound:.2f} – ${self.zopa.upper_bound:.2f} "
                f"(size: ${self.zopa.size:.2f})"
            )
            lines.append(
                f"  Nash solution: ${self.nash.nash_price:.2f} "
                f"(distance: ${self.nash.distance:.2f}, "
                f"{'buyer-favored' if self.nash.buyer_favored else 'seller-favored'})"
            )
            lines.append(
                f"  Pareto efficient: {self.pareto.is_efficient} | "
                f"Egalitarian score: {self.pareto.egalitarian_score:.2f}"
            )
            lines.append(
                f"  Efficiency: {self.efficiency_score:.2f} | Fairness: {self.fairness_score:.2f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict (for the API).

        All float values are sanitized so that ``inf`` and ``nan`` are replaced
        with ``None``, which is safe for ``json.dumps(allow_nan=False)``.
        """

        def _safe(v: float | None) -> float | None:
            """Return v unless it is non-finite (inf / nan), in which case None."""
            if v is None:
                return None
            return v if math.isfinite(v) else None

        return {
            "negotiation_id": self.negotiation_id,
            "outcome": self.outcome,
            "final_price": self.final_price,
            "efficiency_score": round(self.efficiency_score, 4),
            "fairness_score": round(self.fairness_score, 4),
            "summary": self.summary,
            "zopa": {
                "lower_bound": self.zopa.lower_bound,
                "upper_bound": self.zopa.upper_bound,
                "size": round(self.zopa.size, 2),
                "exists": self.zopa.exists,
                "utilization": round(self.zopa.utilization, 4),
            },
            "nash": {
                "nash_price": round(self.nash.nash_price, 2),
                "actual_price": self.nash.actual_price,
                "distance": round(self.nash.distance, 2),
                "distance_pct": round(self.nash.distance_pct, 4),
                "buyer_favored": self.nash.buyer_favored,
            },
            "pareto": {
                "is_efficient": self.pareto.is_efficient,
                "total_surplus": round(self.pareto.total_surplus, 2),
                "buyer_surplus": round(self.pareto.buyer_surplus, 2),
                "seller_surplus": round(self.pareto.seller_surplus, 2),
                "surplus_ratio": _safe(self.pareto.surplus_ratio),
                "egalitarian_score": round(self.pareto.egalitarian_score, 4),
            },
            "batna": {
                "buyer_batna_distance": round(self.batna.buyer_batna_distance, 2),
                "seller_batna_distance": round(self.batna.seller_batna_distance, 2),
                "both_above_batna": self.batna.both_above_batna,
            },
            "concession": {
                "total_rounds": self.concession.total_rounds,
                # first_offer / final_price were computed on the dataclass but
                # never serialised, so the API dropped them.
                "first_offer": round(self.concession.first_offer, 2),
                "final_price": round(self.concession.final_price, 2),
                "buyer_total_concession": round(self.concession.buyer_total_concession, 2),
                "seller_total_concession": round(self.concession.seller_total_concession, 2),
                "buyer_concession_rate": round(self.concession.buyer_concession_rate, 2),
                "seller_concession_rate": round(self.concession.seller_concession_rate, 2),
                "buyer_retraction": round(self.concession.buyer_retraction, 2),
                "seller_retraction": round(self.concession.seller_retraction, 2),
                "attributed": self.concession.attributed,
                "price_trajectory": [round(p, 2) for p in self.concession.price_trajectory],
            },
        }


class GameTheoryAnalyzer:
    """Computes game-theoretic metrics for completed negotiations."""

    def analyze(
        self,
        *,
        negotiation_id: str,
        buyer_max: float,
        seller_min: float,
        initial_price: float,
        final_price: float | None,
        outcome: str,
        rounds_used: int,
        max_rounds: int,
        price_trajectory: list[float] | None = None,
        trajectory_actors: list[str] | None = None,
    ) -> NegotiationGameTheoryReport:
        """Run the full game-theoretic analysis on a completed negotiation.

        Args:
            negotiation_id: Unique identifier of the negotiation.
            buyer_max:       Buyer's maximum budget.
            seller_min:      Seller's minimum acceptable price.
            initial_price:   Seller's opening offer.
            final_price:     Agreed price (None if rejected/timeout).
            outcome:         'accepted', 'rejected', or 'timeout'.
            rounds_used:     Number of rounds completed.
            max_rounds:      Maximum rounds configured.
            price_trajectory: Optional list of prices at each round.
            trajectory_actors: Optional role label for each trajectory entry
                (e.g. ``["initial", "buyer", "seller", ...]``). When supplied,
                concessions are attributed per actor instead of being derived
                from the net initial→final movement.

        Returns:
            NegotiationGameTheoryReport with all analyses populated.
        """
        agreed_price = final_price if (outcome == "accepted" and final_price is not None) else None

        zopa = self._analyze_zopa(buyer_max, seller_min, agreed_price)
        nash = self._analyze_nash(buyer_max, seller_min, agreed_price, zopa)
        pareto = self._analyze_pareto(buyer_max, seller_min, agreed_price)
        batna = self._analyze_batna(buyer_max, seller_min, agreed_price)
        concession = self._analyze_concessions(
            initial_price, agreed_price, rounds_used, price_trajectory, trajectory_actors
        )

        efficiency = self._compute_efficiency(outcome, rounds_used, max_rounds, zopa, pareto)
        fairness = self._compute_fairness(pareto, nash)

        return NegotiationGameTheoryReport(
            negotiation_id=negotiation_id,
            outcome=outcome,
            final_price=agreed_price,
            zopa=zopa,
            nash=nash,
            pareto=pareto,
            batna=batna,
            concession=concession,
            efficiency_score=efficiency,
            fairness_score=fairness,
        )

    # -- ZOPA ---------------------------------------------------------------

    def _analyze_zopa(
        self, buyer_max: float, seller_min: float, final_price: float | None
    ) -> ZOPAAnalysis:
        size = buyer_max - seller_min
        exists = size >= 0
        utilization = 0.0
        if exists and final_price is not None and size > 0:
            # Where in the ZOPA does the final price land? 0 = seller's end, 1 = buyer's end
            utilization = (final_price - seller_min) / size
            utilization = max(0.0, min(1.0, utilization))
        return ZOPAAnalysis(
            lower_bound=seller_min,
            upper_bound=buyer_max,
            size=round(size, 2),
            exists=exists,
            utilization=round(utilization, 4),
        )

    # -- Nash Bargaining Solution -------------------------------------------

    def _analyze_nash(
        self,
        buyer_max: float,
        seller_min: float,
        final_price: float | None,
        zopa: ZOPAAnalysis,
    ) -> NashAnalysis:
        # Symmetric Nash Bargaining Solution = ZOPA midpoint
        nash_price = (buyer_max + seller_min) / 2.0

        if final_price is None:
            return NashAnalysis(
                nash_price=round(nash_price, 2),
                actual_price=0.0,
                distance=0.0,
                distance_pct=0.0,
                buyer_favored=False,
            )

        distance = abs(final_price - nash_price)
        distance_pct = (distance / zopa.size) if zopa.size > 0 else 0.0
        buyer_favored = final_price < nash_price  # buyer pays less than Nash → better for buyer

        return NashAnalysis(
            nash_price=round(nash_price, 2),
            actual_price=round(final_price, 2),
            distance=round(distance, 2),
            distance_pct=round(distance_pct, 4),
            buyer_favored=buyer_favored,
        )

    # -- Pareto Efficiency --------------------------------------------------

    def _analyze_pareto(
        self, buyer_max: float, seller_min: float, final_price: float | None
    ) -> ParetoAnalysis:
        total_surplus = max(0.0, buyer_max - seller_min)

        if final_price is None:
            return ParetoAnalysis(
                is_efficient=False,
                total_surplus=total_surplus,
                buyer_surplus=0.0,
                seller_surplus=0.0,
                surplus_ratio=None,
                egalitarian_score=0.0,
            )

        buyer_surplus = max(0.0, buyer_max - final_price)
        seller_surplus = max(0.0, final_price - seller_min)
        is_efficient = seller_min <= final_price <= buyer_max

        # Use None instead of inf when seller has no surplus (avoids JSON serialisation error).
        surplus_ratio: float | None = None
        if seller_surplus > 1e-9:
            surplus_ratio = round(buyer_surplus / seller_surplus, 4)

        egalitarian_score = 0.0
        if total_surplus > 1e-9:
            egalitarian_score = 1.0 - abs(buyer_surplus - seller_surplus) / total_surplus
            egalitarian_score = max(0.0, egalitarian_score)

        return ParetoAnalysis(
            is_efficient=is_efficient,
            total_surplus=round(total_surplus, 2),
            buyer_surplus=round(buyer_surplus, 2),
            seller_surplus=round(seller_surplus, 2),
            surplus_ratio=surplus_ratio,  # already rounded above, or None
            egalitarian_score=round(egalitarian_score, 4),
        )

    # -- BATNA --------------------------------------------------------------

    def _analyze_batna(
        self, buyer_max: float, seller_min: float, final_price: float | None
    ) -> BATNAAnalysis:
        if final_price is None:
            return BATNAAnalysis(
                buyer_batna_distance=0.0,
                seller_batna_distance=0.0,
                both_above_batna=False,
            )
        buyer_dist = buyer_max - final_price  # positive = buyer saved money vs. BATNA
        seller_dist = final_price - seller_min  # positive = seller earned above BATNA
        both_above = buyer_dist >= 0 and seller_dist >= 0

        return BATNAAnalysis(
            buyer_batna_distance=round(buyer_dist, 2),
            seller_batna_distance=round(seller_dist, 2),
            both_above_batna=both_above,
        )

    # -- Concession Analysis ------------------------------------------------

    def _analyze_concessions(
        self,
        initial_price: float,
        final_price: float | None,
        rounds_used: int,
        trajectory: list[float] | None,
        actors: list[str] | None = None,
    ) -> ConcessionAnalysis:
        price_traj = list(trajectory or [])
        if not price_traj and final_price is not None:
            # Reconstruct minimal trajectory from initial + final
            price_traj = [initial_price, final_price]

        if final_price is None or rounds_used == 0:
            return ConcessionAnalysis(
                total_rounds=rounds_used,
                buyer_total_concession=0.0,
                seller_total_concession=0.0,
                buyer_concession_rate=0.0,
                seller_concession_rate=0.0,
                first_offer=initial_price,
                final_price=final_price or 0.0,
                price_trajectory=price_traj,
            )

        actor_labels = list(actors or [])
        # Attribution needs one label per price point (the actor that produced
        # that price). Without it we can only infer net movement.
        attributed = len(actor_labels) == len(price_traj) and len(price_traj) >= 2

        if attributed:
            # Attribute each move to the agent that made it, measured against
            # that agent's *own* previous offer (not the opponent's price —
            # alternating positions would otherwise be misread as concessions).
            buyer_concession = 0.0
            seller_concession = 0.0
            buyer_retraction = 0.0
            seller_retraction = 0.0
            last_own_price: dict[str, float] = {}
            for i, price in enumerate(price_traj):
                actor = actor_labels[i]
                if actor not in ("buyer", "seller"):
                    continue
                previous = last_own_price.get(actor)
                if previous is not None:
                    delta = price - previous
                    if actor == "buyer":
                        # Buyer paying more = conceding; paying less = retracting.
                        if delta > 0:
                            buyer_concession += delta
                        else:
                            buyer_retraction += -delta
                    else:
                        # Seller accepting less = conceding; asking more = retracting.
                        if delta < 0:
                            seller_concession += -delta
                        else:
                            seller_retraction += delta
                last_own_price[actor] = price
        else:
            # Fallback: net movement from the opening price (monotone assumption).
            buyer_concession = max(0.0, final_price - initial_price)
            seller_concession = max(0.0, initial_price - final_price)
            buyer_retraction = 0.0
            seller_retraction = 0.0

        rate_divisor = max(1, rounds_used)
        return ConcessionAnalysis(
            total_rounds=rounds_used,
            buyer_total_concession=round(buyer_concession, 2),
            seller_total_concession=round(seller_concession, 2),
            buyer_concession_rate=round(buyer_concession / rate_divisor, 2),
            seller_concession_rate=round(seller_concession / rate_divisor, 2),
            first_offer=round(initial_price, 2),
            final_price=round(final_price, 2),
            price_trajectory=[round(p, 2) for p in price_traj],
            buyer_retraction=round(buyer_retraction, 2),
            seller_retraction=round(seller_retraction, 2),
            attributed=attributed,
        )

    # -- Composite Scores ---------------------------------------------------

    def _compute_efficiency(
        self,
        outcome: str,
        rounds_used: int,
        max_rounds: int,
        zopa: ZOPAAnalysis,
        pareto: ParetoAnalysis,
    ) -> float:
        """Efficiency: did they reach a deal, and how quickly?"""
        if outcome != "accepted":
            return 0.0

        # Round efficiency: fewer rounds = better
        round_efficiency = 1.0 - (rounds_used / max(1, max_rounds))

        # ZOPA utilization: closer to midpoint = better
        zopa_quality = 1.0 - abs(zopa.utilization - 0.5) * 2  # 1.0 at midpoint

        # Pareto: did they capture the available surplus?
        pareto_score = 1.0 if pareto.is_efficient else 0.0

        return round((round_efficiency * 0.4 + zopa_quality * 0.3 + pareto_score * 0.3), 4)

    def _compute_fairness(self, pareto: ParetoAnalysis, nash: NashAnalysis) -> float:
        """Fairness: how close to Nash / egalitarian split?"""
        if pareto.total_surplus <= 0:
            return 0.0

        # Egalitarian component (equal surplus split = 1.0)
        egalitarian = pareto.egalitarian_score

        # Nash distance component (close to Nash = fair)
        nash_component = 1.0 - min(1.0, nash.distance_pct * 2)

        return round((egalitarian * 0.6 + nash_component * 0.4), 4)
