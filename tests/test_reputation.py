"""Unit tests for the Reputation Engine."""

from datetime import UTC, datetime, timedelta

from agent_bargain.engine.reputation import NEUTRAL_REPUTATION_SCORE, ReputationEngine
from agent_bargain.models.reputation import FeedbackScore


def test_reputation_no_feedback_returns_none():
    """A brand-new agent (no feedback) must not be scored as 0.0."""
    engine = ReputationEngine()
    assert engine.calculate_score([]) is None
    assert engine.calculate_score([], current_time=datetime.now(UTC)) is None

    record = engine.generate_record("new_agent", [])
    assert record.current_score == NEUTRAL_REPUTATION_SCORE
    assert record.total_transactions == 0


def test_reputation_calculation():
    engine = ReputationEngine(half_life_days=30.0)
    current = datetime.now(UTC)

    # Recent 5-star review (full weight)
    fb1 = FeedbackScore(
        feedback_id="1",
        agreement_id="a1",
        evaluator_id="e1",
        target_id="seller",
        score=5.0,
        timestamp=current,
    )

    # Very old 1-star review (half weight, 30 days old)
    fb2 = FeedbackScore(
        feedback_id="2",
        agreement_id="a2",
        evaluator_id="e2",
        target_id="seller",
        score=1.0,
        timestamp=current - timedelta(days=30),
    )

    # Weight of fb1 is ~1.0
    # Weight of fb2 is ~0.5
    # Weighted sum = (5.0 * 1.0) + (1.0 * 0.5) = 5.5
    # Total weight = 1.0 + 0.5 = 1.5
    # Expected score = 5.5 / 1.5 = 3.666... -> 3.67

    score = engine.calculate_score([fb1, fb2], current_time=current)
    assert 3.6 <= score <= 3.7

    record = engine.generate_record("seller", [fb1, fb2])
    assert record.agent_id == "seller"
    assert record.total_transactions == 2
    assert record.current_score == score


# -- the falsy-zero trap ----------------------------------------------------


def _feedback(score: float, *, days_old: float = 0.0) -> FeedbackScore:
    return FeedbackScore(
        feedback_id=f"fb_{score}_{days_old}",
        agreement_id="a1",
        evaluator_id="e1",
        target_id="seller",
        score=score,
        timestamp=datetime.now(UTC) - timedelta(days=days_old),
    )


def test_fully_decayed_history_counts_as_no_data():
    """Weights that all underflow to 0 leave no usable data: ``None``, then neutral.

    Scores are bounded to 1.0-5.0, so 0.0 was never a real rating — it was an
    artifact that showed a long-dormant agent as the worst one in the market.
    """
    engine = ReputationEngine(half_life_days=30.0)

    # ~100k days old: 0.5 ** (days/30) underflows to exactly 0.0.
    ancient = [_feedback(5.0, days_old=100_000)]
    assert engine.calculate_score(ancient) is None
    assert engine.score_or_neutral(ancient) == NEUTRAL_REPUTATION_SCORE


def test_score_or_neutral_only_substitutes_for_missing_data():
    engine = ReputationEngine()

    assert engine.calculate_score([]) is None
    assert engine.score_or_neutral([]) == NEUTRAL_REPUTATION_SCORE

    # Any real score passes through untouched.
    for score in (1.0, 2.5, 5.0):
        assert engine.score_or_neutral([_feedback(score)]) == score


def test_generate_record_uses_the_same_mapping():
    """``generate_record`` must not reintroduce the ``or`` shortcut."""
    engine = ReputationEngine(half_life_days=30.0)
    ancient = [_feedback(5.0, days_old=100_000)]

    assert engine.generate_record("seller", ancient).current_score == NEUTRAL_REPUTATION_SCORE
    assert engine.generate_record("seller", []).current_score == NEUTRAL_REPUTATION_SCORE
