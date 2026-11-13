"""Evaluator agent behaviour, especially its failure mode.

The reputation ledger is the product's core premise, so an LLM outage must
degrade to *no data*. Persisting a plausible-looking 3.0 instead would be
indistinguishable from a real rating, and exponential decay would keep pulling
an agent's score toward it for weeks.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agent_bargain.agents.evaluator as evaluator_module
import agent_bargain.api.app as app_module
from agent_bargain.llm.schemas import EvaluationResult
from tests.conftest import register_service, start_payload


class _FakeStructured:
    """Like the conftest fake, but able to raise so failure paths are testable."""

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, *args, **kwargs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeRawLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeStructured(self._result)


def _call(**overrides) -> Any:
    """Invoke evaluate_agreement with a representative concluded deal."""
    kwargs: dict[str, Any] = {
        "final_price": 300.0,
        "final_sla": "Standard",
        "history": ["BUYER: ACCEPT - Price: $300.0"],
    }
    kwargs.update(overrides)
    return asyncio.run(evaluator_module.evaluate_agreement(**kwargs))


# -- unit -------------------------------------------------------------------


def test_evaluator_returns_scores_on_success(monkeypatch):
    monkeypatch.setattr(
        evaluator_module,
        "get_evaluation_llm",
        lambda *a, **k: _FakeRawLLM(
            EvaluationResult(
                seller_score=4.5,
                seller_comments="fair",
                buyer_score=3.5,
                buyer_comments="pushy",
            )
        ),
    )

    result = _call()

    assert isinstance(result, EvaluationResult)
    assert result.seller_score == 4.5
    assert result.seller_comments == "fair"
    assert result.buyer_score == 3.5
    assert result.buyer_comments == "pushy"


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("ollama unreachable"),
        ValueError("could not parse structured output"),
    ],
)
def test_evaluator_returns_none_on_failure(monkeypatch, failure):
    """A failed evaluation yields no score at all — never a synthetic 3.0."""
    monkeypatch.setattr(
        evaluator_module, "get_evaluation_llm", lambda *a, **k: _FakeRawLLM(failure)
    )

    assert _call() is None


def test_evaluator_survives_client_construction_failure(monkeypatch):
    """Building the LLM client is inside the guard, not before it.

    Missing or invalid provider config raises during construction; that used to
    escape the node and abort the whole graph after a deal was already reached.
    """

    def _explode(*a, **k):
        raise RuntimeError("google_api_key is empty")

    monkeypatch.setattr(evaluator_module, "get_evaluation_llm", _explode)

    assert _call() is None


# -- integration ------------------------------------------------------------


def test_no_feedback_rows_persisted_when_evaluation_fails(monkeypatch, isolated_settings):
    """An accepted deal with a broken evaluator must write zero feedback rows."""
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: _NullVectorStore())
    monkeypatch.setattr(isolated_settings, "langsmith_tracing", False)

    def _explode(*a, **k):
        raise RuntimeError("evaluator LLM down")

    monkeypatch.setattr(evaluator_module, "get_evaluation_llm", _explode)

    app = app_module.create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        register_service(client)
        response = client.post(
            "/api/v1/negotiations/start",
            json=start_payload(
                engine="strategy",
                buyer_strategy="cooperative",
                seller_strategy="cooperative",
            ),
        )
        assert response.status_code == 200, response.text
        negotiation_id = response.json()["negotiation_id"]

        status = client.get(f"/api/v1/negotiations/{negotiation_id}/status").json()

    # The deal itself still stands — only the evaluation is missing.
    assert status["status"] == "accepted"

    db_file = isolated_settings.sqlite_db_full_path
    conn = sqlite3.connect(db_file)
    try:
        feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        agreement_count = conn.execute("SELECT COUNT(*) FROM agreements").fetchone()[0]
    finally:
        conn.close()

    assert agreement_count == 1, "the agreement should still be recorded"
    assert feedback_count == 0, (
        "a failed evaluation wrote synthetic feedback into the reputation ledger"
    )


class _NullVectorStore:
    """Minimal VectorStore stand-in so no real ChromaDB client is created."""

    def add_services(self, service_ids, documents, metadatas) -> None:
        return None

    def search_services(self, query_text: str, n_results: int = 5) -> dict[str, Any]:
        return {"ids": [[]], "metadatas": [[]]}

    def get_metadatas(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return [None] * len(ids)


# -- ordering: the outcome must be durable before the evaluation runs --------


def test_outcome_is_persisted_before_the_evaluator_runs(monkeypatch, isolated_settings):
    """The terminal status must not wait on a 15-40s LLM call.

    While the evaluator was a graph node on the accepted path, a deal struck in
    ~50ms stayed ``active`` to every client until the evaluation finished. This
    asserts the ordering directly: at the moment the evaluator is invoked, the
    negotiation row is already terminal and the agreement already exists.
    """
    import agent_bargain.engine.negotiation_runner as runner_module

    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: _NullVectorStore())
    monkeypatch.setattr(isolated_settings, "langsmith_tracing", False)

    db_file = isolated_settings.sqlite_db_full_path
    observed: dict[str, Any] = {}

    async def _spy(*, final_price, final_sla, history):
        # Read the same file over a separate connection.
        conn = sqlite3.connect(db_file)
        try:
            observed["status"] = conn.execute("SELECT status FROM negotiations").fetchone()[0]
            observed["agreements"] = conn.execute("SELECT COUNT(*) FROM agreements").fetchone()[0]
            observed["feedback"] = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        finally:
            conn.close()
        return EvaluationResult(
            seller_score=4.0,
            seller_comments="ok",
            buyer_score=4.0,
            buyer_comments="ok",
        )

    monkeypatch.setattr(runner_module, "evaluate_agreement", _spy)

    app = app_module.create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        register_service(client)
        response = client.post(
            "/api/v1/negotiations/start",
            json=start_payload(
                engine="strategy",
                buyer_strategy="cooperative",
                seller_strategy="cooperative",
            ),
        )
        assert response.status_code == 200, response.text

    assert observed, "the evaluator was never called"
    seen_status = observed["status"]
    assert seen_status == "accepted", (
        f"status was still {seen_status!r} when the evaluation started — clients "
        "would see the deal as in-flight for the whole LLM call"
    )
    assert observed["agreements"] == 1
    assert observed["feedback"] == 0, "feedback must not exist before the evaluation"


def test_evaluation_arrives_as_its_own_event(monkeypatch, isolated_settings, offline_llms):
    """Clients learn the outcome first and the scores second, as two events.

    ``offline_llms`` is required, not a convenience: this test builds its own
    app, so without it the evaluation reaches a real Ollama — slow (~45s) and
    the asserted scores become a coin toss.
    """
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: _NullVectorStore())
    monkeypatch.setattr(isolated_settings, "langsmith_tracing", False)

    events: list[dict] = []

    async def _capture(message: str, **_: object) -> None:
        events.append(json.loads(message))

    app = app_module.create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(client.app.state.ws_manager, "broadcast", _capture)
        register_service(client)
        client.post(
            "/api/v1/negotiations/start",
            json=start_payload(
                engine="strategy",
                buyer_strategy="cooperative",
                seller_strategy="cooperative",
            ),
        )

    types = [e["type"] for e in events]
    assert "negotiation_update" in types
    assert "evaluation_update" in types
    assert types.index("negotiation_update") < types.index("evaluation_update"), types

    evaluation = next(e for e in events if e["type"] == "evaluation_update")
    assert evaluation["seller_score"] == 4.0
    assert evaluation["buyer_score"] == 4.0
    # The outcome event no longer carries a score it cannot know yet.
    outcome = next(e for e in events if e["type"] == "negotiation_update")
    assert "evaluation_score" not in outcome
