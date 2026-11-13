"""Reputation system with time decay calculation."""

import math
from datetime import UTC, datetime

from agent_bargain.models.reputation import FeedbackScore, ReputationRecord

#: Score used for agents with no feedback history. A brand-new agent must not
#: be presented to the LLM as a 0.0 (which reads as "worst possible"); a neutral
#: midpoint signals "no data yet".
NEUTRAL_REPUTATION_SCORE = 3.0

#: Summary string used when an agent has no feedback history.
#: Defined once here to avoid copy-paste drift across modules.
DEFAULT_REPUTATION_SUMMARY = "No prior transaction history — this is a new agent."


class ReputationEngine:
    """Calculates agent reputation based on historical feedback with exponential time decay."""

    def __init__(self, half_life_days: float = 30.0):
        """
        Initialize the engine.
        :param half_life_days: Number of days before a feedback score loses half its weight.
        """
        self.half_life_days = half_life_days

    def calculate_score(
        self, feedbacks: list[FeedbackScore], current_time: datetime | None = None
    ) -> float | None:
        """
        Calculate the weighted average score.
        Decay formula: weight = 0.5 ** (days_old / half_life_days)

        Returns None if no feedback data is available (new agent).
        """
        if not feedbacks:
            return None

        if current_time is None:
            current_time = datetime.now(UTC)

        total_weight = 0.0
        weighted_sum = 0.0

        for feedback in feedbacks:
            fb_time = feedback.timestamp
            if fb_time.tzinfo is None:
                fb_time = fb_time.replace(tzinfo=UTC)

            delta = current_time - fb_time
            days_old = max(0.0, delta.total_seconds() / 86400.0)

            weight = math.pow(0.5, days_old / self.half_life_days)

            weighted_sum += feedback.score * weight
            total_weight += weight

        if total_weight == 0.0:
            # Every weight underflowed (feedback centuries old): no usable
            # data, which is ``None`` — not 0.0, the worst possible score.
            return None

        return round(weighted_sum / total_weight, 2)

    def score_or_neutral(
        self, feedbacks: list[FeedbackScore], current_time: datetime | None = None
    ) -> float:
        """``calculate_score`` with the "no data" case mapped to the neutral score.

        Callers must not write ``calculate_score(...) or NEUTRAL_REPUTATION_SCORE``:
        ``calculate_score`` signals "no data" with ``None``, so ``or`` would also
        swallow a legitimate ``0.0`` and flatter it into a 3.0.
        """
        score = self.calculate_score(feedbacks, current_time=current_time)
        return NEUTRAL_REPUTATION_SCORE if score is None else score

    def generate_record(self, agent_id: str, feedbacks: list[FeedbackScore]) -> ReputationRecord:
        """Generate a full reputation record for an agent."""
        return ReputationRecord(
            agent_id=agent_id,
            current_score=self.score_or_neutral(feedbacks),
            total_transactions=len(feedbacks),
        )
