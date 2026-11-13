"""Analytics API routes — game theory metrics and aggregate statistics."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from agent_bargain.analytics.game_theory import GameTheoryAnalyzer
from agent_bargain.api.auth import verify_admin_api_key
from agent_bargain.config import settings
from agent_bargain.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()

_analyzer = GameTheoryAnalyzer()

#: ZOPA/Nash/BATNA analysis is built from the seller's floor and the buyer's
#: budget, and derived metrics (``seller_surplus = final_price - floor``, the Nash
#: midpoint, ...) give them back even if the raw bounds were dropped. Since every
#: regular key can act as any party, these routes are operator-only.
_ADMIN_ONLY = [Depends(verify_admin_api_key)]

#: How many of the most recent accepted deals feed the aggregate
#: efficiency/fairness averages (each needs a full game-theory analysis).
_SCORE_SAMPLE = 50


def _extract_price_trajectory(
    messages: list[dict[str, Any]],
) -> tuple[list[float], list[str]]:
    """Build a role-tagged price trajectory from persisted negotiation messages.

    Each message stores the agent decision as JSON in ``raw_data``; when that
    contains a price we can attribute each price point to the agent that
    produced it. Messages without structured data are skipped.
    """
    prices: list[float] = []
    actors: list[str] = []
    for msg in messages:
        raw = msg.get("raw_data")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        price = data.get("price")
        if price is None:
            continue
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        role = data.get("role") or msg.get("role") or "unknown"
        prices.append(price_value)
        actors.append(str(role))
    return prices, actors


@router.get("/{negotiation_id}/game-theory", dependencies=_ADMIN_ONLY)
async def get_game_theory_analysis(
    negotiation_id: str,
    request: Request,
) -> dict[str, Any]:
    """Return Nash, Pareto, ZOPA, BATNA, and concession analysis for a negotiation."""
    db: DatabaseManager = request.app.state.db
    record = await db.get_negotiation(negotiation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Negotiation {negotiation_id!r} not found.")

    status = record.get("status", "unknown")
    final_price = record.get("final_price")
    rounds_used = record.get("rounds", 0) or 0
    max_budget = record.get("max_budget") or 0.0
    min_price = record.get("min_price") or 0.0
    initial_price = record.get("initial_price") or max_budget

    # Map DB status to outcome
    outcome_map = {"accepted": "accepted", "rejected": "rejected", "timeout": "timeout"}
    outcome = outcome_map.get(status, "unknown")

    # Use the limit this negotiation actually ran with. Falling back to the
    # global setting is only correct for rows written before max_rounds was
    # persisted (NULL); using it unconditionally scored round efficiency
    # against the wrong denominator.
    max_rounds = record.get("max_rounds") or settings.max_negotiation_rounds

    messages = await db.get_messages_for_negotiation(negotiation_id)
    trajectory, actors = _extract_price_trajectory(messages)

    report = _analyzer.analyze(
        negotiation_id=negotiation_id,
        buyer_max=max_budget,
        seller_min=min_price,
        initial_price=initial_price,
        final_price=final_price,
        outcome=outcome,
        rounds_used=rounds_used,
        max_rounds=max_rounds,
        price_trajectory=trajectory or None,
        trajectory_actors=actors if len(actors) == len(trajectory) and trajectory else None,
    )
    return report.to_dict()


# Aggregate averages are admin-only too: over a single accepted negotiation the
# average fairness is that negotiation's fairness, which leaks the floor.
@router.get("/aggregate", dependencies=_ADMIN_ONLY)
async def get_aggregate_analytics(request: Request) -> dict[str, Any]:
    """Aggregate marketplace statistics for the dashboard overview panel."""
    db: DatabaseManager = request.app.state.db
    # Counts and averages come from SQL over every row; only the game-theory
    # scores, which need per-row analysis, use a bounded sample.
    stats = await db.negotiation_stats()
    total = int(stats.get("total") or 0)
    if total == 0:
        return {
            "total_negotiations": 0,
            "accepted": 0,
            "rejected": 0,
            "timeout": 0,
            "acceptance_rate": 0.0,
            "avg_rounds": 0.0,
            "avg_final_price": 0.0,
            "avg_efficiency": 0.0,
            "avg_fairness": 0.0,
        }
    accepted_count = int(stats["accepted"])
    accepted = await db.list_negotiations(limit=_SCORE_SAMPLE, status="accepted")

    # Compute game theory metrics for accepted negotiations
    efficiency_scores = []
    fairness_scores = []

    for neg in accepted:
        try:
            report = _analyzer.analyze(
                negotiation_id=neg["negotiation_id"],
                buyer_max=neg.get("max_budget") or 0.0,
                seller_min=neg.get("min_price") or 0.0,
                initial_price=neg.get("initial_price") or 0.0,
                final_price=neg.get("final_price"),
                outcome="accepted",
                rounds_used=neg.get("rounds", 0) or 0,
                max_rounds=neg.get("max_rounds") or settings.max_negotiation_rounds,
            )
            efficiency_scores.append(report.efficiency_score)
            fairness_scores.append(report.fairness_score)
        except Exception:
            # Aggregate stats stay useful without every row, but a silent
            # ``pass`` hides rows quietly dropping out of the averages.
            logger.warning(
                "Skipped negotiation %s in aggregate analytics",
                neg.get("negotiation_id"),
                exc_info=True,
            )

    return {
        "total_negotiations": total,
        "accepted": accepted_count,
        "rejected": int(stats["rejected"]),
        "timeout": int(stats["timeout"]),
        "acceptance_rate": round(accepted_count / total, 4),
        "avg_rounds": round(float(stats["avg_rounds"]), 2),
        "avg_final_price": round(float(stats["avg_final_price"]), 2),
        # Efficiency/fairness: over the most recent _SCORE_SAMPLE accepted deals.
        "avg_efficiency": round(
            sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0, 4
        ),
        "avg_fairness": round(
            sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0.0, 4
        ),
    }


@router.get("/strategies")
async def list_strategies() -> dict[str, Any]:
    """List all registered negotiation strategies (for the framework/SDK showcase).

    Built-ins are registered when ``agent_bargain.strategies`` is imported, so
    no lazy imports are needed here.
    """
    from agent_bargain.strategies.registry import StrategyRegistry

    return {"strategies": StrategyRegistry.list_all(), "count": len(StrategyRegistry.names())}
