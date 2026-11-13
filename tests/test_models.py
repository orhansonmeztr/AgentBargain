"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from agent_bargain.models.negotiation import NegotiationRecord, NegotiationStatus
from agent_bargain.models.reputation import FeedbackScore


def test_feedback_score_validation():
    # Valid score
    feedback = FeedbackScore(
        feedback_id="fb-1",
        agreement_id="agr-1",
        evaluator_id="buyer-1",
        target_id="seller-1",
        score=4.5,
    )
    assert feedback.score == 4.5

    # Invalid score (should raise ValidationError)
    with pytest.raises(ValidationError):
        FeedbackScore(
            feedback_id="fb-2",
            agreement_id="agr-1",
            evaluator_id="buyer-1",
            target_id="seller-1",
            score=6.0,  # Invalid, max is 5.0
        )


def _record(**overrides):
    payload = {
        "negotiation_id": "neg-1",
        "buyer_id": "b-1",
        "seller_id": "s-1",
        "service_id": "svc-1",
        "rounds": 0,
    }
    payload.update(overrides)
    return NegotiationRecord(**payload)


def test_negotiation_status_is_enforced():
    """``status`` is a validated enum, not a free-form string.

    It used to be declared as ``str`` with an unused ``NegotiationStatus``
    sitting next to it, so any typo became a status the rest of the system
    (analytics outcome mapping, dashboard colouring) silently ignored.
    """
    assert _record().status is NegotiationStatus.active

    for status in NegotiationStatus:
        assert _record(status=status.value).status is status

    with pytest.raises(ValidationError):
        _record(status="unknown")


def test_negotiation_record_max_rounds_optional():
    """``max_rounds`` is nullable so pre-migration rows still load."""
    assert _record().max_rounds is None
    assert _record(max_rounds=3).max_rounds == 3
