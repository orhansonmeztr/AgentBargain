"""Shared turn guardrails and message construction for negotiation agents.

The LLM node (``agents/base.py``), the strategy node (``agents/strategy_node.py``)
and the benchmark runner all produce a turn from a raw decision. Keeping the
rules in one place matters for more than tidiness: a benchmark that skips them
measures an agent that cannot occur in the real graph, so its leaderboard
stops describing the product.

Rules, in order:

0. A non-finite price (``NaN``/``inf``) is never acted on: a ``counter`` or
   ``reject`` carrying one becomes a ``reject`` at the price on the table.
   ``NaN`` compares false against every limit, so letting it through skipped
   the floor/budget checks and could settle a deal at ``NaN``.
1. Prices are never negative.
2. A ``counter`` must stay within the offering party's own limit (a buyer
   never offers above its budget, a seller never below its floor).
3. A ``counter`` never retracts the party's own previous offer: a buyer's bid
   never drops below its last bid, a seller's ask never rises above its last
   ask. An offer, once made, stays on the table — otherwise one side walking
   its price away drags the other after it and they never meet. Rule 2 still
   wins, and the listing price is *not* an offer here (anchoring above it on
   the opening turn is a legitimate tactic).
3a. A ``counter`` that meets or crosses the price on the table becomes an
   ``accept`` of that price. A buyer offering at least what the seller asks
   (or a seller asking at most what the buyer bids) has already agreed —
   countering instead would hand the other side more than it asked for, and
   the price would ratchet past both openings rather than converge between
   them. An unchanged counter is the same case, so it settles instead of
   spinning until timeout.
4. An ``accept`` whose own ``price`` is clearly better *for the decider* than
   the price on the table (a buyer naming less, a seller naming more, by more
   than :data:`ACCEPT_AS_COUNTER_TOLERANCE`) is treated as a ``counter`` at that
   price, then goes through rules 2-3a. The decider said what it wants; the
   action word was wrong. A zero, negative or non-finite price is not taken as
   an intent and leaves the ``accept`` alone.
5. An ``accept`` settles at the price currently on the table (the decider's own
   ``price`` field is ignored, since it may be zero or hallucinated) and is
   downgraded to ``reject`` if that price violates the party's own limit.

Guardrail notes never contain the limit itself. The note lands in the turn's
reasoning, which the opponent reads (its prompt history), and which the
transcript endpoint and the WebSocket stream expose — ``[capped at budget
($500)]`` handed the seller the buyer's exact ceiling.

The opponent also never sees the other side's free-text reasoning: an LLM
routinely writes "my budget is $500" there. :func:`visible_history` renders
the other party's turns as action, price and SLA only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage

from agent_bargain.engine.reputation import DEFAULT_REPUTATION_SUMMARY, NEUTRAL_REPUTATION_SCORE

Role = Literal["buyer", "seller"]


@dataclass(frozen=True)
class GuardedDecision:
    """A decision after the guardrails have been applied."""

    action: str
    price: float
    sla: str
    reasoning: str


#: Relative gap below which an accept's own price counts as "the same price".
#: Echoing $129.99 against a $130 ask is rounding, not a counter-offer, and
#: treating it as one would only add a round.
ACCEPT_AS_COUNTER_TOLERANCE = 0.01


def _wants_better_price(role: Role, price: float, current_price: float) -> bool:
    """True when an accept's ``price`` asks for clearly better terms than the table."""
    if not (math.isfinite(price) and price > 0 and current_price > 0):
        return False
    gap = (current_price - price) if role == "buyer" else (price - current_price)
    return gap > current_price * ACCEPT_AS_COUNTER_TOLERANCE


def apply_guardrails(
    *,
    action: str,
    price: float,
    sla: str,
    reasoning: str,
    role: Role,
    limit: float,
    current_price: float,
    own_last_offer: float | None = None,
) -> GuardedDecision:
    """Clamp a raw decision into one the state machine can safely act on.

    Args:
        action:        ``"accept"``, ``"counter"`` or ``"reject"``.
        price:         Price the decider proposed.
        sla:           SLA the decider proposed.
        reasoning:     Explanation, appended with a ``[Guardrail: ...]`` note
                       whenever a rule fires.
        role:          Which party is deciding.
        limit:         That party's own constraint — ``max_budget`` for a
                       buyer, ``min_price`` for a seller.
        current_price: Price currently on the table.
        own_last_offer: The deciding party's own previous offer in this
                       negotiation (``None`` before its first turn). A counter
                       may not retract from it — see rule 3.
    """
    if not math.isfinite(current_price):
        return GuardedDecision(
            action="reject",
            price=0.0,
            sla=sla,
            reasoning=reasoning + " [Guardrail: no valid price on the table — rejecting]",
        )
    if action != "accept" and not math.isfinite(price):
        return GuardedDecision(
            action="reject",
            price=float(current_price),
            sla=sla,
            reasoning=reasoning + " [Guardrail: proposed price is not a finite number — rejecting]",
        )

    if price < 0:
        price = 0.0
        reasoning += " [Guardrail: Prevented negative price]"

    if action == "accept" and _wants_better_price(role, price, current_price):
        # "Accept" while naming a clearly better price is a counter with the
        # wrong action word — small models do this routinely ("a price closer
        # to $120 is fair" + accept at $130). Settling at the table price
        # would hand the other side the deal this party just argued against.
        action = "counter"
        reasoning += " [Guardrail: accept named a better price — treated as a counter]"

    if action == "counter" and own_last_offer is not None and math.isfinite(own_last_offer):
        # No retracting: an offer once made stays on the table. A seller that
        # raised its ask every round (650 -> 670 -> 680) dragged the buyer up
        # after it and the two never met; holding it at 650 lets the buyer's
        # next bid cross it and settle.
        if role == "buyer" and price < own_last_offer:
            price = float(own_last_offer)
            reasoning += " [Guardrail: bid cannot go below own previous bid]"
        elif role == "seller" and price > own_last_offer:
            price = float(own_last_offer)
            reasoning += " [Guardrail: ask cannot go above own previous ask]"

    if action == "counter":
        # A counter must respect the offering party's own limit. Checking only
        # the *accepting* side let a buyer offer above its budget (or a seller
        # below its floor) and have the counterparty accept it.
        if role == "buyer" and price > limit:
            price = float(limit)
            reasoning += " [Guardrail: counter capped at own budget]"
        elif role == "seller" and price < limit:
            price = float(limit)
            reasoning += " [Guardrail: counter raised to own minimum]"

        crossed = price >= current_price if role == "buyer" else price <= current_price
        if crossed:
            action = "accept"
            reasoning += " [Guardrail: counter meets the price on the table — accepting it]"

    if action == "accept":
        # Settle at the price on the table, not at whatever the decider echoed.
        price = float(current_price)
        if role == "buyer" and price > limit:
            action = "reject"
            reasoning += " [Guardrail: Cannot accept above own budget]"
        elif role == "seller" and price < limit:
            action = "reject"
            reasoning += " [Guardrail: Cannot accept below own minimum]"

    return GuardedDecision(action=action, price=price, sla=sla, reasoning=reasoning)


def _turn_text(role: Role, decision: GuardedDecision, *, with_reasoning: bool) -> str:
    # LLMs often end the SLA with its own period; appending ours made "testing..".
    sla = decision.sla.rstrip().rstrip(".")
    text = f"{role.upper()}: {decision.action.upper()} - Price: ${decision.price}, SLA: {sla}."
    return f"{text} Reasoning: {decision.reasoning}" if with_reasoning else text


def build_turn_message(
    *,
    role: Role,
    decision: GuardedDecision,
    raw_data: dict[str, Any] | None = None,
) -> AIMessage:
    """Render a guarded decision as the graph message both nodes emit.

    ``raw_data`` and ``role`` are what downstream consumers rely on: the
    WebSocket broadcast reads them for the dashboard, persistence stores
    ``raw_data`` as JSON, and the strategy node reconstructs
    ``metadata["price_history"]`` from ``(role, price)`` pairs.
    ``public_content`` is the same turn without the reasoning — what the
    opponent is allowed to read (see :func:`visible_history`).
    """
    return AIMessage(
        content=_turn_text(role, decision, with_reasoning=True),
        additional_kwargs={
            "raw_data": raw_data,
            "role": role,
            "public_content": _turn_text(role, decision, with_reasoning=False),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


def own_last_offer(role: str, messages: Sequence[Any]) -> float | None:
    """``role``'s most recent offered price in the graph messages, if any."""
    for msg in reversed(messages):
        kwargs = getattr(msg, "additional_kwargs", None) or {}
        raw = kwargs.get("raw_data") or {}
        if (kwargs.get("role") or raw.get("role")) == role and raw.get("price") is not None:
            return float(raw["price"])
    return None


def visible_history(viewer: str, turns: Iterable[tuple[str | None, str, str]]) -> list[str]:
    """The transcript as ``viewer`` may read it, from ``(role, content, public)`` turns.

    The viewer's own turns keep their reasoning; the opponent's are reduced to
    action, price and SLA, so neither side can read the other's private limit
    out of its free-text reasoning.
    """
    return [content if role == viewer else public for role, content, public in turns]


def public_text(content: str) -> str:
    """A rendered turn without its reasoning (``"... SLA: x."``)."""
    return content.split(" Reasoning:", 1)[0]


def public_content(msg: Any) -> str:
    """The reasoning-free form of a graph message."""
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    return kwargs.get("public_content") or public_text(str(msg.content))


def history_for(viewer: str, messages: Sequence[Any]) -> list[str]:
    """:func:`visible_history` over graph messages."""
    turns: list[tuple[str | None, str, str]] = []
    for msg in messages:
        kwargs = getattr(msg, "additional_kwargs", None) or {}
        turns.append((kwargs.get("role"), str(msg.content), public_content(msg)))
    return visible_history(viewer, turns)


def opponent_of(role: str) -> Role:
    return "seller" if role == "buyer" else "buyer"


def opponent_reputation(state: dict[str, Any], role: str) -> tuple[float, str]:
    """``(score, summary)`` of the party ``role`` is negotiating against.

    The buyer sees the seller's reputation and vice versa; never its own.
    """
    opponent = opponent_of(role)
    return (
        float(state.get(f"{opponent}_reputation_score", NEUTRAL_REPUTATION_SCORE)),
        str(state.get(f"{opponent}_reputation_summary", DEFAULT_REPUTATION_SUMMARY)),
    )


def turn_update(role: Role, guarded: GuardedDecision) -> dict[str, Any]:
    """The state update a node returns for one guarded turn.

    ``raw_data`` carries the *applied* decision, not the decider's raw output.
    The WebSocket broadcast, the persisted message rows and the game-theory
    price trajectory all read it, so it has to agree with what the state
    machine actually did. Any intervention is still visible: the guardrails
    append a ``[Guardrail: ...]`` note to the reasoning.
    """
    message = build_turn_message(
        role=role,
        decision=guarded,
        raw_data={
            "action": guarded.action,
            "price": guarded.price,
            "sla": guarded.sla,
            "reasoning": guarded.reasoning,
        },
    )
    return {
        "messages": [message],
        "current_price": guarded.price,
        "current_sla": guarded.sla,
        "last_action": guarded.action,
        "last_actor": role,
    }
