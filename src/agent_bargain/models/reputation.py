"""Reputation and feedback models."""

from datetime import datetime

from pydantic import BaseModel, Field

from agent_bargain.models.utils import utc_now


class FeedbackScore(BaseModel):
    """Post-trade evaluation feedback."""

    feedback_id: str
    agreement_id: str
    evaluator_id: str
    target_id: str
    score: float = Field(..., ge=1.0, le=5.0, description="Rating from 1.0 to 5.0")
    evaluator_model: str | None = None
    comments: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class ReputationRecord(BaseModel):
    """Aggregated reputation record for an agent."""

    agent_id: str
    current_score: float
    total_transactions: int
    last_updated: datetime = Field(default_factory=utc_now)
