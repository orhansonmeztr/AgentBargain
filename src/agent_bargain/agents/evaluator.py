"""Post-trade evaluation.

This is *not* a LangGraph node. It used to sit on the accepted path
(``accepted -> evaluator -> END``), which meant the negotiation's terminal
status could not be persisted or broadcast until the evaluation finished — a
deal struck in ~50ms stayed ``active`` to every client for the 15-40s the LLM
took. Evaluation is a consequence of a completed trade, not a state of the
bargaining itself, so the caller now persists the outcome first and evaluates
afterwards.

On failure this returns ``None`` and the caller records no feedback. It
deliberately does not fall back to a neutral 3.0: a fabricated score is
indistinguishable from a real one once persisted, and exponential time decay
would keep dragging a genuinely excellent agent toward it for weeks. An LLM
outage must degrade to *no data*, not to *plausible data*.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from agent_bargain.config import settings
from agent_bargain.llm.prompts import EVALUATOR_PROMPT
from agent_bargain.llm.provider import get_evaluation_llm
from agent_bargain.llm.schemas import EvaluationResult

logger = logging.getLogger(__name__)


async def evaluate_agreement(
    *,
    final_price: float | None,
    final_sla: str | None,
    history: Sequence[str],
) -> EvaluationResult | None:
    """Score both parties for a concluded agreement.

    Async and bounded by ``settings.evaluation_timeout_seconds``. It used to be
    a synchronous call dispatched to a worker thread with no time limit: an
    LLM that never answered pinned that thread (and an A2A task waiting on the
    result) indefinitely, and cancellation could not reach it. ``ainvoke`` is
    cancelled together with the timeout, like the negotiation nodes.

    Args:
        final_price: Agreed price.
        final_sla:   Agreed SLA terms.
        history:     Rendered turn messages, in order.

    Returns:
        The evaluation, or ``None`` if it could not be produced in time. ``None``
        means "no evaluation happened" — it is never a score.
    """
    prompt = EVALUATOR_PROMPT.format_messages(
        agreement_details=f"Final Price: ${final_price}\nFinal SLA: {final_sla}",
        history="\n".join(history),
    )

    try:
        # Constructing the client can fail too (missing/invalid provider
        # config), so it belongs inside the guard, not before it.
        llm = get_evaluation_llm().with_structured_output(EvaluationResult)
        return await asyncio.wait_for(
            llm.ainvoke(prompt), timeout=settings.evaluation_timeout_seconds
        )
    except TimeoutError:
        logger.warning(
            "Evaluation timed out after %ss — recording no feedback for this agreement.",
            settings.evaluation_timeout_seconds,
        )
        return None
    except Exception:
        logger.warning(
            "Evaluation failed — recording no feedback for this agreement "
            "rather than a synthetic score.",
            exc_info=True,
        )
        return None
