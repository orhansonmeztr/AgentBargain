"""Tests for the A2A (Agent-to-Agent) server backed by ``a2a-sdk``.

The unit tests pin the advertised Agent Card; the integration tests drive the
real JSON-RPC endpoint (``SendMessage``) through the app's own ``create_app``
fixture and assert that a task maps to a negotiation, reusing the same offline
LLM stubs as the HTTP negotiation tests.
"""

from __future__ import annotations

import json

from a2a.utils.constants import PROTOCOL_VERSION_1_0

from agent_bargain.protocols.a2a_server import (
    A2A_CARD_URL,
    A2A_RPC_URL,
    AGENT_NAME,
    AGENT_VERSION,
    build_agent_card,
)
from tests.conftest import register_service, start_payload


def _a2a_send(client, payload: dict, message_id: str = "m1"):
    """Send a ``SendMessage`` JSON-RPC request and return the HTTP response."""
    return client.post(
        A2A_RPC_URL,
        headers={"A2A-Version": PROTOCOL_VERSION_1_0},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": json.dumps(payload)}],
                }
            },
        },
    )


def _result_task(response):
    body = response.json()
    assert "error" not in body, body
    return body["result"]["task"]


# -- Agent Card -------------------------------------------------------------


def test_agent_card_advertises_the_negotiation_skill():
    card = build_agent_card()

    assert card.name == AGENT_NAME
    assert card.version == AGENT_VERSION
    assert card.capabilities.streaming is False
    assert card.capabilities.push_notifications is False
    assert "application/json" in card.default_input_modes

    assert len(card.skills) == 1
    skill = card.skills[0]
    assert skill.id == "negotiate"

    assert len(card.supported_interfaces) == 1
    iface = card.supported_interfaces[0]
    assert iface.url == A2A_RPC_URL
    assert iface.protocol_binding == "JSONRPC"
    assert iface.protocol_version == PROTOCOL_VERSION_1_0


def test_agent_card_uses_absolute_rpc_url_when_base_url_is_given():
    card = build_agent_card(public_base_url="https://market.example.com")

    assert card.supported_interfaces[0].url == f"https://market.example.com{A2A_RPC_URL}"


def test_agent_card_is_served(client):
    response = client.get(A2A_CARD_URL)
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == AGENT_NAME
    assert body["version"] == AGENT_VERSION
    assert any(s["id"] == "negotiate" for s in body["skills"])


# -- Task execution ---------------------------------------------------------


def test_send_message_runs_a_negotiation(client):
    register_service(client)

    response = _a2a_send(
        client,
        start_payload(
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
        ),
    )
    assert response.status_code == 200, response.text

    task = _result_task(response)
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"

    artifacts = task["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "negotiation_result"
    data = artifacts[0]["parts"][0]["data"]
    assert data["status"] == "accepted"
    assert data["negotiation_id"].startswith("neg_")
    assert data["final_price"] is not None


def test_send_message_invalid_payload_reports_a_failed_task(client):
    response = _a2a_send(client, {"this": "is not a negotiation request"})
    assert response.status_code == 200, response.text

    task = _result_task(response)
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    # The failure text is carried on the terminal status message.
    message_text = task["status"]["message"]["parts"][0]["text"]
    assert "Invalid negotiation request" in message_text


def test_send_message_unregistered_service_reports_a_failed_task(client):
    response = _a2a_send(
        client,
        {
            "buyer_id": "b1",
            "seller_id": "s1",
            "service_id": "missing_svc",
            "max_budget": 500.0,
            "engine": "strategy",
            "buyer_strategy": "cooperative",
            "seller_strategy": "cooperative",
        },
    )
    assert response.status_code == 200, response.text

    task = _result_task(response)
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    message_text = task["status"]["message"]["parts"][0]["text"]
    assert "not registered" in message_text
