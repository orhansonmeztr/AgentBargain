"""A2A authentication, the exposed-host startup guard and settings bounds."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import agent_bargain.api.app as app_module
from agent_bargain.api import auth
from agent_bargain.config import Settings, settings
from agent_bargain.protocols.a2a_server import A2A_CARD_URL, A2A_RPC_URL, build_agent_card
from tests.conftest import AUTH, KEY, register_service, start_payload


def _send(client, headers: dict | None = None):
    from a2a.utils.constants import PROTOCOL_VERSION_1_0

    return client.post(
        A2A_RPC_URL,
        headers={"A2A-Version": PROTOCOL_VERSION_1_0, **(headers or {})},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": "m1",
                    "role": "ROLE_USER",
                    "parts": [
                        {
                            "text": json.dumps(
                                start_payload(
                                    engine="strategy",
                                    buyer_strategy="cooperative",
                                    seller_strategy="cooperative",
                                )
                            )
                        }
                    ],
                }
            },
        },
    )


# -- A2A JSON-RPC requires a key when auth is on -----------------------------


def test_a2a_rpc_without_key_is_401(auth_client):
    response = _send(auth_client)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "X-API-Key"


def test_a2a_rpc_with_wrong_key_is_401(auth_client):
    assert _send(auth_client, {"X-API-Key": "nope"}).status_code == 401


@pytest.mark.parametrize("headers", [AUTH, {"Authorization": f"Bearer {KEY}"}])
def test_a2a_rpc_with_valid_key_runs(auth_client, headers):
    register_service(auth_client, headers=AUTH)
    response = _send(auth_client, headers)
    assert response.status_code == 200, response.text
    assert "error" not in response.json()


def test_a2a_card_stays_public_and_advertises_the_scheme(auth_client):
    response = auth_client.get(A2A_CARD_URL)
    assert response.status_code == 200
    body = response.json()
    scheme = body["securitySchemes"]["apiKey"]["apiKeySecurityScheme"]
    assert scheme["name"] == "X-API-Key"
    assert scheme["location"] == "header"


def test_a2a_open_when_auth_disabled(client):
    register_service(client)
    assert _send(client).status_code == 200


def test_card_has_no_scheme_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    assert len(build_agent_card().security_schemes) == 0


# -- exposed host without keys is refused ------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"])
def test_loopback_hosts_are_allowed_without_keys(monkeypatch, host):
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    monkeypatch.setattr(settings, "allow_unauthenticated", False)
    auth.require_auth_when_exposed(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_exposed_host_without_keys_is_refused(monkeypatch, host):
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    monkeypatch.setattr(settings, "allow_unauthenticated", False)
    with pytest.raises(RuntimeError, match="Refusing to start"):
        auth.require_auth_when_exposed(host)


def test_exposed_host_allowed_with_keys(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", [KEY])
    auth.require_auth_when_exposed("0.0.0.0")


def test_exposed_host_allowed_with_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    monkeypatch.setattr(settings, "allow_unauthenticated", True)
    auth.require_auth_when_exposed("0.0.0.0")


def test_app_refuses_to_start_exposed_without_keys(
    isolated_settings, monkeypatch, fake_vector_store
):
    monkeypatch.setattr(app_module, "create_vector_store", lambda *a, **k: fake_vector_store)
    monkeypatch.setattr(isolated_settings, "server_host", "0.0.0.0")
    app = app_module.create_app()
    with pytest.raises(RuntimeError, match="Refusing to start"), TestClient(app):
        pass


def test_default_host_is_loopback():
    assert Settings.model_fields["server_host"].default == "127.0.0.1"


# -- settings bounds -----------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_negotiation_rounds", 0),
        ("max_negotiation_rounds", 1001),
        ("negotiation_timeout_seconds", 0),
        ("evaluation_timeout_seconds", 0),
    ],
)
def test_out_of_range_settings_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
