"""Negotiators' reasoning is admin-only, and the evaluator never sees it.

LLM reasoning routinely states the party's own limit ("my budget is $300",
"my minimum acceptable price is $80"). A regular key can act as any party, so
it must not read that text — the same rule that hides ``min_price`` and
``max_budget``. The evaluator gets the public transcript too, because its
comments become the reputation summary the *next* counterparty reads.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import agent_bargain.engine.negotiation_runner as runner_module
from agent_bargain.api.routes.websocket import ConnectionManager
from tests.conftest import ADMIN, AUTH, register_service, start_payload

_TURN = {
    "type": "agent_activity",
    "negotiation_id": "neg_1",
    "action": "counter",
    "price": 250.0,
    "reasoning": "my budget is $300",
    "content": "BUYER: COUNTER - Price: $250.0, SLA: S. Reasoning: my budget is $300",
    "public_content": "BUYER: COUNTER - Price: $250.0, SLA: S.",
}


def _socket() -> MagicMock:
    ws = MagicMock()
    ws.send_text = AsyncMock()
    return ws


def _sent(ws: MagicMock) -> dict:
    return json.loads(ws.send_text.await_args.args[0])


# -- WebSocket fan-out --------------------------------------------------------


async def test_turn_events_are_redacted_for_regular_connections():
    manager = ConnectionManager()
    admin, regular = _socket(), _socket()
    manager.active_connections.extend([admin, regular])
    manager.reasoning_readers.add(admin)

    await manager.broadcast_event(_TURN)

    full, public = _sent(admin), _sent(regular)
    assert full["reasoning"] == "my budget is $300"
    assert "Reasoning:" in full["content"]
    assert public["reasoning"] == ""
    assert public["content"] == "BUYER: COUNTER - Price: $250.0, SLA: S."
    assert "300" not in json.dumps(public)
    assert "public_content" not in full and "public_content" not in public


async def test_other_events_reach_everyone_unchanged():
    manager = ConnectionManager()
    admin, regular = _socket(), _socket()
    manager.active_connections.extend([admin, regular])
    manager.reasoning_readers.add(admin)
    event = {"type": "negotiation_update", "negotiation_id": "n", "status": "accepted"}

    await manager.broadcast_event(event)

    assert _sent(admin) == _sent(regular) == event


def test_disconnect_forgets_the_reader():
    manager = ConnectionManager()
    ws = _socket()
    manager.active_connections.append(ws)
    manager.reasoning_readers.add(ws)
    manager.disconnect(ws)
    assert ws not in manager.reasoning_readers


def _first_turn(ws) -> dict:
    while True:
        event = ws.receive_json()
        if event["type"] == "agent_activity":
            return event


def _strategy_payload() -> dict:
    return start_payload(
        engine="strategy", buyer_strategy="cooperative", seller_strategy="cooperative"
    )


def test_live_stream_redacts_for_a_regular_key(auth_client):
    register_service(auth_client, headers=AUTH)
    with auth_client.websocket_connect(f"/api/v1/ws/events?api_key={AUTH['X-API-Key']}") as ws:
        auth_client.post("/api/v1/negotiations/start", json=_strategy_payload(), headers=AUTH)
        turn = _first_turn(ws)
    assert turn["reasoning"] == ""
    assert "Reasoning:" not in turn["content"]


def test_live_stream_is_complete_for_an_admin_key(auth_client):
    register_service(auth_client, headers=AUTH)
    with auth_client.websocket_connect(f"/api/v1/ws/events?api_key={ADMIN['X-API-Key']}") as ws:
        auth_client.post("/api/v1/negotiations/start", json=_strategy_payload(), headers=AUTH)
        turn = _first_turn(ws)
    assert turn["reasoning"]
    assert "Reasoning:" in turn["content"]


# -- transcript endpoint -------------------------------------------------------


def _transcript(client, headers) -> list[dict]:
    neg = client.post("/api/v1/negotiations/start", json=_strategy_payload(), headers=AUTH).json()[
        "negotiation_id"
    ]
    response = client.get(f"/api/v1/negotiations/{neg}/messages", headers=headers)
    assert response.status_code == 200
    return response.json()["messages"]


def test_transcript_hides_reasoning_from_a_regular_key(auth_client):
    register_service(auth_client, headers=AUTH)
    messages = _transcript(auth_client, AUTH)
    assert messages
    for m in messages:
        assert "Reasoning:" not in m["content"]
        assert m["content"].endswith(".")
        assert "reasoning" not in json.loads(m["raw_data"])
        assert json.loads(m["raw_data"])["price"] is not None  # the turn itself survives


def test_transcript_is_complete_for_an_admin_key(auth_client):
    register_service(auth_client, headers=AUTH)
    messages = _transcript(auth_client, ADMIN)
    assert all("Reasoning:" in m["content"] for m in messages)
    assert all("reasoning" in json.loads(m["raw_data"]) for m in messages)


def test_transcript_is_complete_when_auth_is_off(client):
    register_service(client)
    neg = client.post("/api/v1/negotiations/start", json=_strategy_payload()).json()[
        "negotiation_id"
    ]
    messages = client.get(f"/api/v1/negotiations/{neg}/messages").json()["messages"]
    assert all("Reasoning:" in m["content"] for m in messages)


# -- evaluator input -----------------------------------------------------------


def test_evaluator_receives_the_public_transcript(client, monkeypatch):
    seen: dict = {}

    async def _spy(*, final_price, final_sla, history):
        seen["history"] = list(history)
        return None

    monkeypatch.setattr(runner_module, "evaluate_agreement", _spy)
    register_service(client)
    client.post("/api/v1/negotiations/start", json=_strategy_payload())

    assert seen["history"], "the evaluator was not called"
    assert all("Reasoning:" not in line for line in seen["history"])
    assert all(line.startswith(("BUYER:", "SELLER:")) for line in seen["history"])
