"""Negotiation models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agent_bargain.models.utils import utc_now


class NegotiationStatus(StrEnum):
    """Lifecycle states of a negotiation record."""

    active = "active"
    accepted = "accepted"
    rejected = "rejected"
    timeout = "timeout"
    failed = "failed"


class NegotiationRequest(BaseModel):
    """Request to open a negotiation over an already-registered service.

    The seller's side of the deal — their floor price, listing title and SLA —
    comes from the registered listing (``POST /api/v1/marketplace/services``),
    not from this body. Only the *buyer's* own constraints belong here. Letting
    the caller supply ``min_price`` meant the buyer could declare the seller's
    walk-away point, send ``min_price=0`` and win every negotiation.

    Shared by the HTTP ``POST /negotiations/start`` route and the A2A server's
    ``negotiate`` task, so both surfaces exercise identical validation.
    """

    buyer_id: str = Field(min_length=1)
    #: Must match the seller on the registered listing; a mismatch is rejected
    #: rather than silently renegotiated against someone else's service.
    seller_id: str = Field(min_length=1)
    #: Must already be registered.
    service_id: str = Field(min_length=1)
    max_budget: float = Field(ge=0.0, description="The buyer's own ceiling")
    #: Opening offer. Defaults to the listing's ``base_price``; if supplied it
    #: must not undercut the seller's registered floor.
    initial_price: float | None = Field(default=None, ge=0.0)
    #: Defaults to the listing's ``sla_terms``.
    initial_sla: str | None = None
    #: Round limit for this negotiation. ``None`` -> ``settings.max_negotiation_rounds``.
    #: Persisted, so analytics can score round efficiency against the real limit.
    max_rounds: int | None = Field(default=None, ge=1, le=1000)
    engine: Literal["llm", "strategy"] = "llm"
    buyer_strategy: str | None = None  # required when engine="strategy"
    seller_strategy: str | None = None  # required when engine="strategy"

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> NegotiationRequest:
        if self.engine == "strategy":
            if not self.buyer_strategy:
                raise ValueError("buyer_strategy is required when engine='strategy'")
            if not self.seller_strategy:
                raise ValueError("seller_strategy is required when engine='strategy'")
            # Eagerly validate that both strategy names are registered.
            from agent_bargain.strategies.registry import StrategyRegistry, UnknownStrategyError

            try:
                StrategyRegistry.get(self.buyer_strategy)
                StrategyRegistry.get(self.seller_strategy)
            except UnknownStrategyError as exc:
                raise ValueError(str(exc)) from exc
        return self


class NegotiationMessage(BaseModel):
    """A message exchanged during negotiation."""

    message_id: str
    negotiation_id: str
    role: str
    content: str
    raw_data: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class Agreement(BaseModel):
    """A finalized agreement between buyer and seller."""

    agreement_id: str
    buyer_id: str
    seller_id: str
    service_id: str
    final_price: float
    final_sla: str
    timestamp: datetime = Field(default_factory=utc_now)


class NegotiationRecord(BaseModel):
    """A record of a negotiation attempt (successful or failed)."""

    negotiation_id: str
    buyer_id: str
    seller_id: str
    service_id: str
    max_budget: float | None = None
    min_price: float | None = None
    initial_price: float | None = None
    status: NegotiationStatus = NegotiationStatus.active
    rounds: int
    #: Round limit this negotiation actually ran with. Persisted because
    #: round-efficiency analytics needs it; ``None`` means "global default".
    max_rounds: int | None = None
    final_price: float | None = None
    final_sla: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
