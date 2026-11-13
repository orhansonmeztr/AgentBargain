"""
CLI entry point for the benchmark suite.

Usage:
    uv run python -m agent_bargain.benchmarks
    uv run python -m agent_bargain.benchmarks --strategies aggressive,cooperative --reps 3
"""

from __future__ import annotations

import argparse

from agent_bargain.benchmarks.runner import BenchmarkRunner
from agent_bargain.benchmarks.scenarios import SCENARIOS
from agent_bargain.strategies.registry import StrategyRegistry


# ANSI colors
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


def print_header():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║              ⚖️  AgentBargain Benchmark Suite                ║
║         Strategy vs Strategy — Game Theory Metrics           ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def print_leaderboard(report):
    lb = report.leaderboard()
    print(
        f"{C.BOLD}{C.CYAN}{'Rank':<6}{'Strategy':<16}{'Score':>8}{'Win%':>8}{'Eff':>8}{'Fair':>8}{'Avg$':>10}{'AvgRnd':>8}{C.RESET}"
    )
    print(f"{C.DIM}{'─' * 72}{C.RESET}")

    medals = ["🥇", "🥈", "🥉"]
    for i, s in enumerate(lb):
        medal = medals[i] if i < 3 else f"  {i + 1}."
        color = [C.GREEN, C.YELLOW, C.MAGENTA][i] if i < 3 else C.DIM
        print(
            f"{color}{medal:<6}{s.name:<16}"
            f"{s.score:>7.3f} "
            f"{s.win_rate * 100:>6.1f}% "
            f"{s.avg_efficiency:>7.3f} "
            f"{s.avg_fairness:>7.3f} "
            f"{'$' + f'{s.avg_surplus:.0f}':>9} "
            f"{s.avg_rounds:>7.1f}"
            f"{C.RESET}"
        )


def print_match_details(report, top_n=10):
    accepted = [m for m in report.matches if m.outcome == "accepted"]
    rejected = [m for m in report.matches if m.outcome != "accepted"]

    print(f"\n{C.BOLD}📊 Match Summary{C.RESET}")
    print(f"   Total matches: {len(report.matches)}")
    print(f"   {C.GREEN}Accepted: {len(accepted)}{C.RESET}")
    print(f"   {C.RED}Rejected/Timeout: {len(rejected)}{C.RESET}")
    print(f"   Elapsed: {report.elapsed_seconds:.2f}s\n")

    # Show most interesting matches
    if accepted:
        # Sort by Nash distance (most interesting = furthest from Nash)
        interesting = sorted(accepted, key=lambda m: m.report.nash.distance, reverse=True)[:top_n]
        print(f"{C.BOLD}🎯 Most Interesting Deals (furthest from Nash):{C.RESET}")
        print(
            f"{C.DIM}{'Buyer':<14}{'Seller':<14}{'Scenario':<16}{'Price':>8}{'Nash':>8}{'Δ':>8}{'Pareto':>8}{C.RESET}"
        )
        print(f"{C.DIM}{'─' * 74}{C.RESET}")
        for m in interesting:
            r = m.report
            pareto = f"{C.GREEN}✓{C.RESET}" if r.pareto.is_efficient else f"{C.RED}✗{C.RESET}"
            favor = f"{C.CYAN}B" if r.nash.buyer_favored else f"{C.MAGENTA}S"
            print(
                f"  {m.buyer_strategy:<14}{m.seller_strategy:<14}{m.scenario:<16}"
                f"${m.final_price:>7.0f}${r.nash.nash_price:>7.0f}"
                f"  {favor}{r.nash.distance:>5.0f}{C.RESET}"
                f"  {pareto}"
            )


def print_scenario_breakdown(report):
    print(f"\n{C.BOLD}📋 Per-Scenario Breakdown:{C.RESET}")
    scenarios = {}
    for m in report.matches:
        scenarios.setdefault(m.scenario, []).append(m)

    for name, matches in sorted(scenarios.items()):
        accepted = sum(1 for m in matches if m.outcome == "accepted")
        total = len(matches)
        rate = accepted / total * 100 if total else 0
        color = C.GREEN if rate > 60 else C.YELLOW if rate > 30 else C.RED
        print(f"   {name:<20} {color}{accepted:>3}/{total:>3} accepted ({rate:>5.1f}%){C.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="AgentBargain Benchmark Suite — pit strategies against each other."
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategy names (default: all registered)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Repetitions per matchup (default: 1, increase for stochastic strategies)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated list of scenario names (default: all)",
    )
    args = parser.parse_args()

    print_header()

    # Parse strategy list
    strategy_names = None
    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
        print(f"  Strategies: {', '.join(strategy_names)}")
    else:
        strategy_names = StrategyRegistry.names()
        print(f"  Strategies: {', '.join(strategy_names)} (all)")

    # Parse scenario list
    scenario_list = None
    if args.scenarios:
        from agent_bargain.benchmarks.scenarios import get_scenario

        scenario_list = [get_scenario(s.strip()) for s in args.scenarios.split(",")]
        print(f"  Scenarios:  {', '.join(s.name for s in scenario_list)}")
    else:
        scenario_list = SCENARIOS
        print(f"  Scenarios:  {', '.join(s.name for s in SCENARIOS)} (all)")

    total_matchups = len(strategy_names) ** 2 * len(scenario_list) * args.reps
    print(f"  Matchups:   {total_matchups}")
    print()

    # Run benchmark
    runner = BenchmarkRunner()
    report = runner.run(
        strategies=strategy_names,
        scenarios=scenario_list,
        repetitions=args.reps,
    )

    # Output results
    print(f"\n{C.BOLD}{C.CYAN}🏆 Strategy Leaderboard{C.RESET}\n")
    print_leaderboard(report)
    print_match_details(report)
    print_scenario_breakdown(report)

    print(f"\n{C.DIM}Done in {report.elapsed_seconds:.2f}s.{C.RESET}\n")


if __name__ == "__main__":
    main()
