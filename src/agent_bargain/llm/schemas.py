"""Structured output schemas for LLM responses.

Using Pydantic models with ChatOllama.with_structured_output() enforces
JSON Schema at the Ollama level — eliminating parse failures.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NegotiationDecision(BaseModel):
    """Structured output for buyer/seller negotiation agents."""

    action: Literal["accept", "counter", "reject"] = Field(
        description="The action taken: accept the offer, counter with new terms, or reject outright."
    )
    price: float = Field(
        description="The price being proposed or agreed to. Must be a positive number.",
        # NaN compares false against every limit; reject it at parse time so
        # the fallback LLM gets a chance instead.
        allow_inf_nan=False,
    )
    sla: str = Field(description="The SLA (Service Level Agreement) being proposed or agreed to.")
    reasoning: str = Field(description="Brief explanation of why this decision was made.")


class EvaluationResult(BaseModel):
    """Structured output for the evaluator agent — rates both parties."""

    seller_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Performance score for the seller, from 1.0 (poor) to 5.0 (excellent).",
    )
    seller_comments: str = Field(description="Short explanation justifying the seller score.")
    buyer_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Performance score for the buyer, from 1.0 (poor) to 5.0 (excellent).",
    )
    buyer_comments: str = Field(description="Short explanation justifying the buyer score.")
