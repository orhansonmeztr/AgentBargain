"""WebSocket streaming and the app-scoped ConnectionManager.

The manager used to be a module-level singleton shared by every ``create_app()``,
so events from one application leaked to clients of another. These tests pin the
per-app scoping and exercise the connect/broadcast/disconnect path, which had no
coverage at all.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import agent_bargain.api.app as app_module
from agent_bargain.api.routes.websocket import ConnectionManager
from tests.conftest import register_service, start_payload

# -- app scoping ----------------------------------------------------


def test_each_app_gets_its_own_manager(isolated_settings, monkeypatch, fake_vector_store):
    """Two applications must not share one connection list."""
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: fake_vector_store)
    monkeypatch.setattr(isolated_settings, "langsmith_tracing", False)

    with TestClient(app_module.create_app()) as a, TestClient(app_module.create_app()) as b:
        assert a.app.state.ws_manager is not b.app.state.ws_manager
        assert isinstance(a.app.state.ws_manager, ConnectionManager)


def test_manager_is_not_a_module_singleton():
    """The old import-time global must be gone, or scoping is only cosmetic."""
    import agent_bargain.api.routes.websocket as ws_module

    assert not hasattr(ws_module, "manager"), (
        "module-level `manager` singleton is back — apps would share connections"
    )


# -- ConnectionManager unit behaviour --------------------------------


async def test_broadcast_reaches_every_connected_client():
    manager = ConnectionManager()
    first = MagicMock()
    first.send_text = AsyncMock()
    second = MagicMock()
    second.send_text = AsyncMock()

    manager.active_connections.extend([first, second])
    await manager.broadcast("hello")

    first.send_text.assert_awaited_once_with("hello")
    second.send_text.assert_awaited_once_with("hello")
    assert len(manager.active_connections) == 2


async def test_broadcast_prunes_dead_connections():
    """A client that vanished mid-send must be dropped, not break the loop."""
    manager = ConnectionManager()
    healthy = MagicMock()
    healthy.send_text = AsyncMock()
    dead = MagicMock()
    dead.send_text = AsyncMock(side_effect=RuntimeError("connection closed"))

    manager.active_connections.extend([dead, healthy])
    await manager.broadcast("hello")

    assert manager.active_connections == [healthy]
    healthy.send_text.assert_awaited_once_with("hello")


async def test_broadcast_survives_concurrent_list_mutation():
    """Iterating a live list while another coroutine appends would raise."""
    manager = ConnectionManager()
    existing = MagicMock()
    existing.send_text = AsyncMock()
    manager.active_connections.append(existing)

    async def _mutate_during_send(_message):
        # Simulates another negotiation broadcasting / a client connecting
        # while this broadcast is awaiting.
        newcomer = MagicMock()
        newcomer.send_text = AsyncMock()
        manager.active_connections.append(newcomer)

    existing.send_text.side_effect = _mutate_during_send

    await manager.broadcast("hello")  # must not raise RuntimeError


def test_disconnect_is_idempotent():
    manager = ConnectionManager()
    conn = MagicMock()
    manager.active_connections.append(conn)

    manager.disconnect(conn)
    manager.disconnect(conn)  # already gone

    assert manager.active_connections == []


# -- end to end over a real socket ----------------------------------


def test_client_receives_live_events(client):
    """A connected dashboard must see turns as they happen, then the outcome."""
    register_service(client)

    with client.websocket_connect("/api/v1/ws/events") as ws:
        response = client.post(
            "/api/v1/negotiations/start",
            json=start_payload(
                engine="strategy",
                buyer_strategy="aggressive",
                seller_strategy="aggressive",
            ),
        )
        assert response.status_code == 200, response.text
        negotiation_id = response.json()["negotiation_id"]

        # Drain until the terminal event; the number of turns before it depends
        # on how the two strategies bargain, so it must not be hardcoded.
        events: list[dict] = []
        for _ in range(50):
            event = json.loads(ws.receive_text())
            events.append(event)
            if event["type"] == "negotiation_update":
                break
        else:
            pytest.fail(f"no negotiation_update after {len(events)} events")

    types = [e["type"] for e in events]
    assert types.count("agent_activity") >= 2, types

    update = events[-1]
    assert update["negotiation_id"] == negotiation_id
    # str(status) — a StrEnum leaking into JSON would serialise oddly.
    assert isinstance(update["status"], str)
    assert update["status"] in {"accepted", "rejected", "timeout"}
