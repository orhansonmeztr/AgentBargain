"""Shared pytest fixtures for AgentBargain test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import agent_bargain.agents.base as base_module
import agent_bargain.agents.evaluator as evaluator_module
import agent_bargain.api.app as app_module
from agent_bargain.llm.schemas import EvaluationResult, NegotiationDecision
from agent_bargain.storage.vector_store import VectorStoreManager


class FakeStructured:
    def __init__(self, result):
        self._result = result

    def invoke(self, *args, **kwargs):
        return self._result

    async def ainvoke(self, *args, **kwargs):
        return self._result


class FakeRawLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        return FakeStructured(self._result)


@pytest.fixture(autouse=True)
def _no_credentials_from_env(monkeypatch):
    """Every test starts with no keys, whatever the developer's ``.env`` holds.

    ``isolated_settings`` already does this, but only for tests that request
    it. Unit tests that patch just ``api_keys`` still read ``admin_api_keys``
    from ``.env`` — so adding an admin key locally failed two auth tests that
    pass in CI. Tests that need keys set them explicitly afterwards.
    """
    from agent_bargain.config import settings

    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    monkeypatch.setattr(settings, "allow_unauthenticated", False)


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    """Neutralise every setting that a developer's local ``.env`` could leak in.

    ``Settings`` reads ``<project root>/.env``, which is gitignored and differs
    per machine. Patching only the storage paths is not enough — this suite used
    to change behaviour depending on that file:

    * ``api_keys`` set ⇒ every request 401s and the whole suite fails, while CI
      (no ``.env``) stays green.
    * ``langsmith_tracing`` true ⇒ the lifespan exports LANGSMITH_* and real
      runs start posting traces over the network, which also makes them slow.

    So all three are pinned here, and individual tests opt back in explicitly
    (see ``auth_client``, which sets its own ``api_keys``).
    """
    from agent_bargain.config import settings

    monkeypatch.setattr(settings, "sqlite_db_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    monkeypatch.setattr(settings, "langsmith_api_key", "")
    # A local .env with SERVER_HOST=0.0.0.0 and no keys would make every
    # app refuse to start.
    monkeypatch.setattr(settings, "server_host", "127.0.0.1")
    monkeypatch.setattr(settings, "allow_unauthenticated", False)
    return settings


@pytest.fixture
def fake_vector_store():
    """A MagicMock satisfying the VectorStore protocol, so no real ChromaDB."""
    store = MagicMock(spec=VectorStoreManager)
    store.search_services.return_value = {"ids": [[]], "metadatas": [[]]}
    store.get_metadatas.side_effect = lambda ids: [{"title": i} for i in ids]
    return store


@pytest.fixture
def offline_llms(monkeypatch):
    """Stub out every LLM call so tests never reach a model provider.

    Without this a test that runs the post-trade evaluator blocks on a real
    connection attempt to Ollama and only finishes when it times out — measured
    at 43s for a single negotiation before this fixture was shared.
    """
    monkeypatch.setattr(
        base_module,
        "get_negotiation_llm",
        lambda *a, **k: FakeRawLLM(
            NegotiationDecision(action="reject", price=0.0, sla="Standard", reasoning="test")
        ),
    )
    monkeypatch.setattr(
        evaluator_module,
        "get_evaluation_llm",
        lambda *a, **k: FakeRawLLM(
            EvaluationResult(
                seller_score=4.0,
                seller_comments="ok",
                buyer_score=4.0,
                buyer_comments="ok",
            )
        ),
    )


@pytest.fixture
def client(isolated_settings, monkeypatch, fake_vector_store, offline_llms):
    """A TestClient with isolated storage and offline, deterministic LLMs."""
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: fake_vector_store)
    # Never export LangSmith tracing settings from a developer's local .env.
    monkeypatch.setattr(isolated_settings, "langsmith_tracing", False)

    app = app_module.create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


#: The listing most tests negotiate over. ``floor_price`` is the seller's
#: walk-away point and can only be set here, by the seller — never by a buyer
#: through POST /negotiations/start.
DEFAULT_SERVICE: dict[str, object] = {
    "service_id": "srv_1",
    "seller_id": "seller_1",
    "title": "Test Service",
    "description": "A service used by the test suite.",
    "base_price": 400.0,
    "floor_price": 100.0,
    "sla_terms": "Standard",
}


def register_service(
    client: TestClient,
    headers: dict | None = None,
    **overrides,
) -> dict[str, object]:
    """Register a service listing and return the payload that was sent.

    ``headers`` carries the API key for tests whose app has authentication on.
    """
    payload = {**DEFAULT_SERVICE, **overrides}
    response = client.post("/api/v1/marketplace/services", json=payload, headers=headers)
    assert response.status_code == 201, response.text
    return payload


def start_payload(**overrides) -> dict[str, object]:
    """A valid ``POST /negotiations/start`` body against ``DEFAULT_SERVICE``.

    Deliberately contains no ``min_price``: the seller's floor comes from the
    registered listing, not from the caller.
    """
    payload: dict[str, object] = {
        "buyer_id": "buyer_1",
        "seller_id": DEFAULT_SERVICE["seller_id"],
        "service_id": DEFAULT_SERVICE["service_id"],
        "max_budget": 500.0,
        "initial_price": 400.0,
    }
    payload.update(overrides)
    return payload


KEY = "test-key-abc123"
AUTH = {"X-API-Key": KEY}
ADMIN_KEY = "admin-key-xyz789"
ADMIN = {"X-API-Key": ADMIN_KEY}


@pytest.fixture
def auth_client(isolated_settings, monkeypatch, fake_vector_store, offline_llms):
    """A client whose app has exactly one valid API key configured.

    ``offline_llms`` matters here, not just for speed: without it the evaluator
    tries to reach a real Ollama and the test blocks on the connection timeout.
    """
    monkeypatch.setattr(isolated_settings, "api_keys", [KEY])
    monkeypatch.setattr(isolated_settings, "admin_api_keys", [ADMIN_KEY])
    monkeypatch.setattr(isolated_settings, "langsmith_tracing", False)
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: fake_vector_store)
    with TestClient(app_module.create_app(), raise_server_exceptions=False) as c:
        yield c
