"""End-to-end integration tests for the API layer."""

import json

from fastapi.testclient import TestClient

from tests.conftest import register_service, start_payload


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_start_negotiation(client):
    """Test the negotiation start endpoint returns correct structure.

    The background negotiation runs with a mocked LLM, so it completes
    offline and deterministically within the test.
    """
    register_service(client)

    response = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(initial_price=300.0),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert data["negotiation_id"].startswith("neg_")


def test_search_services(client):
    """Test the marketplace search endpoint."""
    response = client.post(
        "/api/v1/marketplace/search",
        json={"query": "web development", "top_k": 5},
    )
    assert response.status_code == 200
    assert "results" in response.json()


def test_strategy_engine_negotiation_completes(client):
    """engine='strategy' must run a deterministic, LLM-free negotiation end-to-end."""
    register_service(client)

    start = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
        ),
    )
    assert start.status_code == 200
    negotiation_id = start.json()["negotiation_id"]

    # TestClient runs background tasks synchronously, so the record must exist now.
    status = client.get(f"/api/v1/negotiations/{negotiation_id}/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "accepted"
    assert body["final_price"] is not None
    assert body["rounds"] >= 1

    messages = client.get(f"/api/v1/negotiations/{negotiation_id}/messages").json()["messages"]
    assert len(messages) >= 2
    assert {m["role"] for m in messages} >= {"buyer", "seller"}


def test_strategy_engine_requires_strategy_names(client):
    """engine='strategy' without strategy names is rejected with 422."""
    register_service(client)

    response = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(engine="strategy"),
    )
    assert response.status_code == 422


def test_strategy_engine_rejects_unknown_strategy(client):
    """Unknown strategy names are rejected at validation time."""
    register_service(client)

    response = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="does_not_exist",
            seller_strategy="cooperative",
        ),
    )
    assert response.status_code == 422


def test_unknown_negotiation_status_is_404(client):
    """Unknown negotiation IDs must 404 rather than masquerade as 'active'."""
    response = client.get("/api/v1/negotiations/neg_missing/status")
    assert response.status_code == 404


def test_start_persists_active_record_before_background_task(client, monkeypatch):
    """A negotiation must be queryable while it is still running (not 404)."""
    register_service(client)

    import agent_bargain.api.routes.negotiations as negotiations_module

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(negotiations_module, "_run_negotiation", _noop)

    start = client.post("/api/v1/negotiations/start", json=start_payload())
    assert start.status_code == 200
    negotiation_id = start.json()["negotiation_id"]

    status = client.get(f"/api/v1/negotiations/{negotiation_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] == "active"


def test_game_theory_analysis_is_json_serializable(client):
    """Full analytics pipeline on a persisted negotiation must not emit inf/nan."""
    # The edge case is a seller whose floor equals the buyer's ceiling, so the
    # deal lands exactly on the floor and seller_surplus is 0. The floor now
    # comes from the registered listing, not from the start request.
    register_service(client, floor_price=500.0, base_price=500.0)

    start = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
            initial_price=500.0,
            max_budget=500.0,
        ),
    )
    assert start.status_code == 200, start.text
    negotiation_id = start.json()["negotiation_id"]
    response = client.get(f"/api/v1/analytics/{negotiation_id}/game-theory")
    assert response.status_code == 200
    body = response.json()
    assert "concession" in body
    assert body["pareto"]["surplus_ratio"] is None


def test_wall_clock_timeout_preserves_reached_agreement(client, monkeypatch):
    """If the deal is reached but post-processing hangs, keep 'accepted'."""
    register_service(client)

    import asyncio

    import agent_bargain.engine.negotiation_runner as runner_module

    class _SlowGraph:
        async def astream(self, state):
            yield {"accepted": {"status": "accepted", "current_price": 300.0}}
            await asyncio.sleep(5)

    monkeypatch.setattr(runner_module, "create_strategy_graph", lambda *a, **k: _SlowGraph())
    monkeypatch.setattr(runner_module.settings, "negotiation_timeout_seconds", 0.1)

    start = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
        ),
    )
    negotiation_id = start.json()["negotiation_id"]

    status = client.get(f"/api/v1/negotiations/{negotiation_id}/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "accepted"
    assert body["final_price"] == 300.0


def test_wall_clock_timeout_marks_still_running_negotiation_as_timeout(client, monkeypatch):
    """A negotiation that never reached a terminal state becomes 'timeout'."""
    register_service(client)

    import asyncio

    import agent_bargain.engine.negotiation_runner as runner_module

    class _HangingGraph:
        async def astream(self, state):
            await asyncio.sleep(5)
            if False:  # pragma: no cover - makes this an async generator
                yield {}

    monkeypatch.setattr(runner_module, "create_strategy_graph", lambda *a, **k: _HangingGraph())
    monkeypatch.setattr(runner_module.settings, "negotiation_timeout_seconds", 0.1)

    start = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
        ),
    )
    negotiation_id = start.json()["negotiation_id"]

    status = client.get(f"/api/v1/negotiations/{negotiation_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] == "timeout"


def test_background_task_reuses_app_db_connection(client, monkeypatch):
    """The background task must be handed ``app.state.db``, not build its own.

    A private ``DatabaseManager()`` is not merely connection churn: with
    ``SQLITE_DB_PATH=:memory:`` every new connection gets its own empty
    database, so every write from the task would silently disappear.
    """
    register_service(client)

    import agent_bargain.api.routes.negotiations as negotiations_module

    captured: dict[str, object] = {}

    async def _capture(state, negotiation_id, db, ws_manager):
        captured["db"] = db
        captured["ws_manager"] = ws_manager

    monkeypatch.setattr(negotiations_module, "_run_negotiation", _capture)

    response = client.post("/api/v1/negotiations/start", json=start_payload())
    assert response.status_code == 200

    assert captured["db"] is client.app.state.db, (
        "background task opened its own database connection"
    )
    assert captured["ws_manager"] is client.app.state.ws_manager, (
        "background task did not receive the app-scoped connection manager"
    )


def test_negotiation_completes_with_in_memory_sqlite(
    monkeypatch, fake_vector_store, isolated_settings
):
    """End-to-end run against ``:memory:`` — the configuration CI declares.

    Regression test: this used to persist nothing, because the background task
    connected to a *different* in-memory database than ``app.state.db``.

    Takes ``isolated_settings`` even though it overrides the DB path: that
    fixture is also what neutralises ``api_keys`` and LangSmith tracing, which
    would otherwise leak in from a developer's local ``.env``.
    """
    import agent_bargain.agents.evaluator as evaluator_module
    import agent_bargain.api.app as app_module
    from agent_bargain.llm.schemas import EvaluationResult
    from tests.conftest import FakeRawLLM

    monkeypatch.setattr(isolated_settings, "sqlite_db_path", ":memory:")
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: fake_vector_store)
    # Keep the post-trade evaluation offline and deterministic.
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

    app = app_module.create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        register_service(c)
        start = c.post(
            "/api/v1/negotiations/start",
            json=start_payload(
                engine="strategy",
                buyer_strategy="cooperative",
                seller_strategy="cooperative",
            ),
        )
        assert start.status_code == 200
        negotiation_id = start.json()["negotiation_id"]

        status = c.get(f"/api/v1/negotiations/{negotiation_id}/status")
        assert status.status_code == 200, "record vanished — background task used another DB"
        assert status.json()["status"] == "accepted"

        # Feedback written by the task must also be visible to the app connection.
        agents = c.get("/api/v1/marketplace/agents").json()["agents"]
        assert {a["agent_id"] for a in agents} >= {"buyer_1", "seller_1"}


def test_max_rounds_is_persisted_and_used_by_analytics(client):
    """A per-negotiation round limit must survive to the analytics endpoint."""
    register_service(client)

    start = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
            max_rounds=3,
        ),
    )
    assert start.status_code == 200
    negotiation_id = start.json()["negotiation_id"]

    analysis = client.get(f"/api/v1/analytics/{negotiation_id}/game-theory")
    assert analysis.status_code == 200
    # Round efficiency is scored against the stored limit, not the global one.
    assert analysis.json()["concession"]["total_rounds"] <= 3


def test_max_rounds_rejects_invalid_values(client):
    """max_rounds is validated at the boundary."""
    register_service(client)

    assert (
        client.post("/api/v1/negotiations/start", json=start_payload(max_rounds=0)).status_code
        == 422
    )
    assert (
        client.post("/api/v1/negotiations/start", json=start_payload(max_rounds=1001)).status_code
        == 422
    )


def test_agent_activity_events_carry_the_round_number(client, monkeypatch):
    """Each streamed turn must say which round it happened in.

    The dashboard plots price against round. Before the payload carried
    ``round``, the client only learned the count from the terminal
    ``negotiation_update`` event, so every live point was plotted at x=0 and
    the whole price timeline collapsed onto one spot.
    """
    register_service(client)

    captured: list[dict] = []

    async def _capture(message: str, **_: object) -> None:
        captured.append(json.loads(message))

    monkeypatch.setattr(client.app.state.ws_manager, "broadcast", _capture)

    # Two aggressive strategies grind instead of settling on the first turn.
    response = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            engine="strategy",
            buyer_strategy="aggressive",
            seller_strategy="aggressive",
        ),
    )
    assert response.status_code == 200

    activity = [e for e in captured if e["type"] == "agent_activity"]
    assert len(activity) >= 2, "expected a multi-turn negotiation"

    assert all("round" in e for e in activity), "agent_activity missing 'round'"
    rounds = [e["round"] for e in activity]
    assert max(rounds) > 0, f"every turn reported round 0: {rounds}"
    assert rounds == sorted(rounds), f"rounds went backwards: {rounds}"


def test_history_does_not_expose_either_partys_limit(client):
    """``min_price`` is the seller's floor and ``max_budget`` the buyer's ceiling.

    Listing responses already hide ``floor_price``; the history list used to
    hand both limits back via ``SELECT *``.
    """
    register_service(client)
    assert client.post("/api/v1/negotiations/start", json=start_payload()).status_code == 200

    rows = client.get("/api/v1/negotiations/history").json()["negotiations"]
    assert rows, "expected the negotiation in the history"
    for row in rows:
        assert "min_price" not in row
        assert "max_budget" not in row


def test_llm_client_construction_failure_does_not_strand_negotiation(client, monkeypatch):
    """A provider that fails while *building* the client must degrade to reject."""
    import agent_bargain.agents.base as base_module

    def _boom(*args, **kwargs):
        raise RuntimeError("missing API key")

    monkeypatch.setattr(base_module, "get_negotiation_llm", _boom)
    monkeypatch.setattr(base_module, "get_fallback_llm", _boom)

    register_service(client)
    neg_id = client.post("/api/v1/negotiations/start", json=start_payload()).json()[
        "negotiation_id"
    ]
    assert client.get(f"/api/v1/negotiations/{neg_id}/status").json()["status"] == "rejected"


def test_graph_crash_is_persisted_as_failed(client, monkeypatch):
    """An exception escaping the graph must not leave the row ``active``."""
    import agent_bargain.engine.negotiation_runner as runner_module

    class _CrashingGraph:
        async def astream(self, state):
            raise RuntimeError("node blew up")
            yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(runner_module, "create_negotiation_graph", lambda: _CrashingGraph())

    register_service(client)
    neg_id = client.post("/api/v1/negotiations/start", json=start_payload()).json()[
        "negotiation_id"
    ]
    record = client.get(f"/api/v1/negotiations/{neg_id}/status").json()
    assert record["status"] == "failed"
