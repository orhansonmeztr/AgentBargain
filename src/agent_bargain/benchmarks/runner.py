"""
Benchmark runner — simulates strategy-vs-strategy negotiations.

Runs each buyer-strategy × seller-strategy × scenario combination,
records game-theoretic metrics, and produces a ranked leaderboard.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field

from agent_bargain.agents.guardrails import (
    apply_guardrails,
    build_turn_message,
    visible_history,
)
from agent_bargain.analytics.game_theory import GameTheoryAnalyzer, NegotiationGameTheoryReport
from agent_bargain.benchmarks.scenarios import SCENARIOS, Scenario
from agent_bargain.engine.reputation import (
    DEFAULT_REPUTATION_SUMMARY,
    NEUTRAL_REPUTATION_SCORE,
)
from agent_bargain.strategies.base import NegotiationContext
from agent_bargain.strategies.estimation import limits_for
from agent_bargain.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single strategy-vs-strategy negotiation."""

    buyer_strategy: str
    seller_strategy: str
    scenario: str
    outcome: str  # accepted, rejected, timeout
    final_price: float | None
    rounds_used: int
    report: NegotiationGameTheoryReport


@dataclass
class StrategyStats:
    """Aggregated stats for a single strategy across all matchups."""

    name: str
    total_matches: int = 0
    wins: int = 0  # accepted deals
    losses: int = 0  # rejected / timeout
    total_surplus: float = 0.0
    total_efficiency: float = 0.0
    total_fairness: float = 0.0
    total_rounds: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.total_matches)

    @property
    def avg_surplus(self) -> float:
        return self.total_surplus / max(1, self.wins)

    @property
    def avg_efficiency(self) -> float:
        return self.total_efficiency / max(1, self.wins)

    @property
    def avg_fairness(self) -> float:
        return self.total_fairness / max(1, self.wins)

    @property
    def avg_rounds(self) -> float:
        return self.total_rounds / max(1, self.total_matches)

    @property
    def score(self) -> float:
        """Composite score: 40% win rate + 30% efficiency + 30% fairness."""
        return self.win_rate * 0.4 + self.avg_efficiency * 0.3 + self.avg_fairness * 0.3


@dataclass
class BenchmarkReport:
    """Full benchmark report with all match results and leaderboard."""

    matches: list[MatchResult] = field(default_factory=list)
    strategy_stats: dict[str, StrategyStats] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def leaderboard(self) -> list[StrategyStats]:
        """Return strategies ranked by composite score (descending)."""
        return sorted(self.strategy_stats.values(), key=lambda s: s.score, reverse=True)


class BenchmarkRunner:
    """Simulates strategy matchups across scenarios and collects metrics."""

    def __init__(self) -> None:
        self._analyzer = GameTheoryAnalyzer()

    def run(
        self,
        strategies: list[str] | None = None,
        scenarios: list[Scenario] | None = None,
        repetitions: int = 1,
    ) -> BenchmarkReport:
        """Run the full benchmark suite.

        Args:
            strategies: List of strategy names to benchmark. None = all registered.
            scenarios:  List of scenarios to test. None = all built-in scenarios.
            repetitions: How many times to repeat each matchup (for stochastic strategies).

        Returns:
            BenchmarkReport with all results and leaderboard.
        """
        if strategies is None:
            strategies = StrategyRegistry.names()
        if scenarios is None:
            scenarios = SCENARIOS

        report = BenchmarkReport()
        start = time.monotonic()

        # Initialize stats for each strategy
        for name in strategies:
            report.strategy_stats[name] = StrategyStats(name=name)

        # Run every buyer×seller×scenario combination
        matchups = list(itertools.product(strategies, strategies, scenarios))
        total = len(matchups) * repetitions

        for idx, (buyer_name, seller_name, scenario) in enumerate(matchups):
            for rep in range(repetitions):
                match_num = idx * repetitions + rep + 1
                logger.debug(
                    "[%d/%d] %s (buyer) vs %s (seller) — %s",
                    match_num,
                    total,
                    buyer_name,
                    seller_name,
                    scenario.name,
                )
                result = self._run_match(buyer_name, seller_name, scenario)
                report.matches.append(result)

                # Update buyer stats
                bs = report.strategy_stats[buyer_name]
                bs.total_matches += 1
                bs.total_rounds += result.rounds_used
                if result.outcome == "accepted":
                    bs.wins += 1
                    bs.total_surplus += result.report.pareto.buyer_surplus
                    bs.total_efficiency += result.report.efficiency_score
                    bs.total_fairness += result.report.fairness_score
                else:
                    bs.losses += 1

                # Update seller stats
                ss = report.strategy_stats[seller_name]
                ss.total_matches += 1
                ss.total_rounds += result.rounds_used
                if result.outcome == "accepted":
                    ss.wins += 1
                    ss.total_surplus += result.report.pareto.seller_surplus
                    ss.total_efficiency += result.report.efficiency_score
                    ss.total_fairness += result.report.fairness_score
                else:
                    ss.losses += 1

        report.elapsed_seconds = time.monotonic() - start
        return report

    def _run_match(
        self,
        buyer_name: str,
        seller_name: str,
        scenario: Scenario,
    ) -> MatchResult:
        """Simulate a single negotiation between two strategies.

        This deliberately mirrors ``create_strategy_node`` turn for turn: same
        context fields (including ``metadata["price_history"]`` and
        ``message_history``), same guardrails, same round accounting. A
        benchmark that skips any of those measures an agent that cannot occur
        in the real graph — which is exactly how ``tit_for_tat`` came to be
        scored on a mirroring rate it never actually sees in production.
        """
        buyer = StrategyRegistry.instantiate(buyer_name)
        seller = StrategyRegistry.instantiate(seller_name)

        price = scenario.initial_price
        sla = "Standard"
        trajectory = [price]
        # Actor label for each trajectory point; the opening price has no actor.
        trajectory_actors = ["initial"]
        # What the node feeds strategies: role-tagged prices and rendered
        # message text for every turn so far.
        price_history: list[tuple[str, float]] = []
        # (role, content, public_content) per turn; each role reads it through
        # visible_history, exactly as the strategy node does.
        turns: list[tuple[str | None, str, str]] = []

        outcome = "timeout"
        final_price = None
        rounds_used = 0

        for round_num in range(scenario.max_rounds):
            settled = False
            for role, strategy, limit in (
                ("buyer", buyer, scenario.buyer_max),
                ("seller", seller, scenario.seller_min),
            ):
                # Same information boundary as the strategy node: the true
                # own limit, an estimate of the opponent's.
                max_budget, min_price = limits_for(
                    role,
                    own_limit=limit,
                    initial_price=scenario.initial_price,
                    price_history=price_history,
                )
                context = NegotiationContext(
                    role=role,
                    max_budget=max_budget,
                    min_price=min_price,
                    current_price=price,
                    current_sla=sla,
                    initial_price=scenario.initial_price,
                    round=round_num,
                    max_rounds=scenario.max_rounds,
                    # Benchmarks have no feedback history, so both parties see
                    # the same neutral score a brand-new agent gets in the API.
                    opponent_reputation=NEUTRAL_REPUTATION_SCORE,
                    opponent_reputation_summary=DEFAULT_REPUTATION_SUMMARY,
                    message_history=visible_history(role, turns),
                    metadata={"price_history": list(price_history)},
                )

                decision = strategy.decide(context)
                guarded = apply_guardrails(
                    action=decision.action,
                    price=decision.price,
                    sla=decision.sla,
                    reasoning=decision.reasoning,
                    role=role,
                    limit=limit,
                    current_price=price,
                    # Same no-retraction input the graph nodes pass.
                    own_last_offer=next(
                        (p for actor, p in reversed(price_history) if actor == role), None
                    ),
                )
                # Rendered through the same helper the graph nodes use, so
                # message_history is byte-identical to what a real negotiation
                # would have shown the strategy.
                message = build_turn_message(role=role, decision=guarded, raw_data=None)
                public = message.additional_kwargs["public_content"]

                # Every turn the graph emits becomes a message row, and
                # analytics builds its trajectory from those rows — *including*
                # the settling turn, whose price is the one on the table.
                trajectory.append(guarded.price)
                trajectory_actors.append(role)
                price_history.append((role, guarded.price))
                turns.append((role, str(message.content), public))

                if role == "seller":
                    # The graph routes seller -> round_counter unconditionally,
                    # so a seller's accept/reject still increments the round.
                    # A buyer settling short-circuits before round_counter.
                    rounds_used = round_num + 1

                if guarded.action == "accept":
                    outcome = "accepted"
                    final_price = guarded.price
                    settled = True
                    break
                if guarded.action == "reject":
                    outcome = "rejected"
                    settled = True
                    break

                price = guarded.price
                sla = guarded.sla

            if settled:
                break

        # Game theory analysis
        neg_id = f"bench_{buyer_name}_vs_{seller_name}_{scenario.name}"
        report = self._analyzer.analyze(
            negotiation_id=neg_id,
            buyer_max=scenario.buyer_max,
            seller_min=scenario.seller_min,
            initial_price=scenario.initial_price,
            final_price=final_price,
            outcome=outcome,
            rounds_used=rounds_used,
            max_rounds=scenario.max_rounds,
            price_trajectory=trajectory,
            trajectory_actors=trajectory_actors,
        )

        return MatchResult(
            buyer_strategy=buyer_name,
            seller_strategy=seller_name,
            scenario=scenario.name,
            outcome=outcome,
            final_price=final_price,
            rounds_used=rounds_used,
            report=report,
        )
