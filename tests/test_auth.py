"""API key authentication.

Auth is *disabled* when ``API_KEYS`` is empty so local development needs no
setup; these tests cover both modes, because the disabled path is the one that
silently exposes the whole API if someone deploys without noticing.
"""

from __future__ import annotations

import hmac

import pytest
from starlette.websockets import WebSocketDisconnect

from agent_bargain.api import auth
from agent_bargain.config import settings
from agent_bargain.protocols.a2a_server import A2A_CARD_URL, AGENT_NAME
from tests.conftest import (
    ADMIN,
    ADMIN_KEY,
    AUTH,
    DEFAULT_SERVICE,
    KEY,
    register_service,
    start_payload,
)

# -- mode switching -------------------------------------------------


def test_auth_disabled_when_no_keys_configured(client):
    """Empty API_KEYS means open access — the documented development default."""
    assert auth.auth_enabled() is False
    assert client.get("/api/v1/marketplace/services").status_code == 200


def test_auth_enabled_when_keys_configured(auth_client):
    assert auth.auth_enabled() is True


def test_isolated_settings_neutralises_env_api_keys(monkeypatch, isolated_settings):
    """A developer's local ``.env`` must not switch auth on for the whole suite.

    Regression: adding a real ``API_KEYS`` to ``.env`` made every test 401
    locally while CI — which has no ``.env`` — stayed green. The same leak
    applies to ``LANGSMITH_TRACING``, which would otherwise make tests post
    traces over the network.
    """
    from agent_bargain.config import Settings

    monkeypatch.setenv("API_KEYS", '["leaked-from-env"]')
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    # A freshly built Settings really would pick the environment up...
    assert Settings().api_keys == ["leaked-from-env"]
    # ...but the fixture pins the singleton the application actually reads.
    assert isolated_settings.api_keys == []
    assert isolated_settings.langsmith_tracing is False
    assert isolated_settings.langsmith_api_key == ""
    assert auth.auth_enabled() is False


# -- enforcement ----------------------------------------------------


def test_health_stays_public(auth_client):
    """Container health checks cannot present a key."""
    assert auth_client.get("/health").status_code == 200


def test_agent_card_stays_public(auth_client):
    """A2A discovery is the card's purpose; hiding it behind a secret defeats it."""
    response = auth_client.get(A2A_CARD_URL)
    assert response.status_code == 200
    assert response.json()["name"] == AGENT_NAME


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/marketplace/services",
        "/api/v1/marketplace/agents",
        "/api/v1/negotiations/history",
        "/api/v1/analytics/aggregate",
        "/api/v1/analytics/strategies",
        "/api/v1/analytics/neg_x/game-theory",
    ],
)
def test_protected_routes_reject_anonymous(auth_client, path):
    response = auth_client.get(path)
    assert response.status_code == 401, f"{path} was reachable without a key"
    assert response.headers.get("WWW-Authenticate") == auth.API_KEY_HEADER


def test_wrong_key_is_rejected(auth_client):
    assert (
        auth_client.get("/api/v1/marketplace/services", headers={"X-API-Key": "nope"}).status_code
        == 401
    )


def test_header_key_is_accepted(auth_client):
    assert auth_client.get("/api/v1/marketplace/services", headers=AUTH).status_code == 200


def test_bearer_token_is_accepted(auth_client):
    headers = {"Authorization": f"Bearer {KEY}"}
    assert auth_client.get("/api/v1/marketplace/services", headers=headers).status_code == 200


def test_non_bearer_authorization_is_rejected(auth_client):
    """A Basic/other scheme must not be mistaken for a key."""
    headers = {"Authorization": f"Basic {KEY}"}
    assert auth_client.get("/api/v1/marketplace/services", headers=headers).status_code == 401


def test_query_key_is_rejected_on_http(auth_client):
    """``?api_key=`` is for the WebSocket only; on HTTP it just leaks into logs."""
    assert auth_client.get(f"/api/v1/marketplace/services?api_key={KEY}").status_code == 401


# -- writes ---------------------------------------------------------


def test_write_endpoints_are_protected(auth_client):
    """Starting a negotiation writes to the reputation ledger, so it must be gated."""
    register = auth_client.post("/api/v1/marketplace/services", json=dict(DEFAULT_SERVICE))
    assert register.status_code == 401

    start = auth_client.post("/api/v1/negotiations/start", json=start_payload())
    assert start.status_code == 401


def test_full_flow_works_with_a_key(auth_client):
    """Registration and negotiation both succeed once authenticated."""
    register = auth_client.post(
        "/api/v1/marketplace/services", headers=AUTH, json=dict(DEFAULT_SERVICE)
    )
    assert register.status_code == 201, register.text

    start = auth_client.post(
        "/api/v1/negotiations/start",
        headers=AUTH,
        json=start_payload(
            engine="strategy", buyer_strategy="cooperative", seller_strategy="cooperative"
        ),
    )
    assert start.status_code == 200, start.text

    negotiation_id = start.json()["negotiation_id"]
    status = auth_client.get(f"/api/v1/negotiations/{negotiation_id}/status", headers=AUTH)
    assert status.json()["status"] == "accepted"


def test_websocket_handshake_is_refused_without_a_key(auth_client):
    """The handshake must be rejected *before* accept(), not after.

    Once a WebSocket is accepted it can no longer be refused with an HTTP 401,
    so an unauthenticated client would otherwise stay connected to the event
    stream and receive every negotiation's live turns.
    """
    with pytest.raises(WebSocketDisconnect), auth_client.websocket_connect("/api/v1/ws/events"):
        pass

    # A wrong key is refused too, not only a missing one.
    with (
        pytest.raises(WebSocketDisconnect),
        auth_client.websocket_connect("/api/v1/ws/events?api_key=wrong"),
    ):
        pass


def test_websocket_connects_with_a_query_key(auth_client):
    register_service(auth_client, headers=AUTH)

    with auth_client.websocket_connect(f"/api/v1/ws/events?api_key={KEY}") as ws:
        response = auth_client.post(
            "/api/v1/negotiations/start",
            headers=AUTH,
            json=start_payload(
                engine="strategy", buyer_strategy="aggressive", seller_strategy="aggressive"
            ),
        )
        assert response.status_code == 200, response.text
        assert "negotiation_id" in response.json()
        # At least one event must arrive over the authenticated socket.
        assert ws.receive_json()["type"] in {"agent_activity", "negotiation_update"}


# -- key comparison -------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(KEY, True), ("wrong", False), ("", False), (None, False), (KEY + "x", False)],
)
def test_matches_is_exact(monkeypatch, candidate, expected):
    monkeypatch.setattr(settings, "api_keys", [KEY])
    assert auth._matches(candidate) is expected


def test_matches_uses_constant_time_comparison(monkeypatch):
    """A plain ``==`` would leak the key's length and matching prefix via timing."""
    monkeypatch.setattr(settings, "api_keys", [KEY])
    calls: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def _spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", _spy)

    assert auth._matches(KEY) is True
    assert calls, "compare_digest was not used"


def test_every_configured_key_is_checked(monkeypatch):
    """Comparison must not short-circuit, or timing reveals which key matched."""
    # The match is first: a generator inside any() would stop right here.
    monkeypatch.setattr(settings, "api_keys", [KEY, "second", "third"])
    calls: list[bytes] = []
    real = hmac.compare_digest

    def _spy(a, b):
        calls.append(b)
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", _spy)

    assert auth._matches(KEY) is True
    assert len(calls) == 3, f"short-circuited after {len(calls)} comparison(s)"


# -- startup warning ------------------------------------------------


def test_warns_when_unauthenticated_on_a_public_interface(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(settings, "api_keys", [])
    with caplog.at_level(logging.WARNING, logger=auth.__name__):
        auth.warn_if_unauthenticated("0.0.0.0")

    assert "UNAUTHENTICATED" in caplog.text or "anyone able to reach this port" in caplog.text


def test_no_warning_when_keys_configured(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(settings, "api_keys", [KEY])
    monkeypatch.setattr(settings, "admin_api_keys", [ADMIN_KEY])
    with caplog.at_level(logging.WARNING, logger=auth.__name__):
        auth.warn_if_unauthenticated("0.0.0.0")

    assert caplog.text == ""


# -- admin keys -----------------------------------------------------
#
# Game-theory analysis is built from the seller's floor and the buyer's budget,
# and every regular key can act as any party, so it is operator-only.

ADMIN_ONLY = ["/api/v1/analytics/aggregate", "/api/v1/analytics/{neg}/game-theory"]


def _finished_negotiation(client) -> str:
    register_service(client, headers=AUTH)
    response = client.post("/api/v1/negotiations/start", json=start_payload(), headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()["negotiation_id"]


@pytest.mark.parametrize("template", ADMIN_ONLY)
def test_regular_key_is_forbidden_from_admin_analytics(auth_client, template):
    path = template.format(neg=_finished_negotiation(auth_client))
    response = auth_client.get(path, headers=AUTH)
    assert response.status_code == 403, f"{path} leaked private limits to a regular key"


@pytest.mark.parametrize("template", ADMIN_ONLY)
def test_admin_key_reaches_admin_analytics(auth_client, template):
    path = template.format(neg=_finished_negotiation(auth_client))
    assert auth_client.get(path, headers=ADMIN).status_code == 200


def test_admin_key_also_works_on_regular_routes(auth_client):
    """One admin key is enough for the whole dashboard."""
    assert auth_client.get("/api/v1/marketplace/services", headers=ADMIN).status_code == 200
    with auth_client.websocket_connect(f"/api/v1/ws/events?api_key={ADMIN_KEY}"):
        pass


def test_strategy_list_stays_on_regular_keys(auth_client):
    assert auth_client.get("/api/v1/analytics/strategies", headers=AUTH).status_code == 200


def test_admin_analytics_fail_closed_without_admin_keys(auth_client, monkeypatch):
    """Auth on but no admin key configured: nobody gets in, not every key."""
    neg = _finished_negotiation(auth_client)
    monkeypatch.setattr(settings, "admin_api_keys", [])
    response = auth_client.get(f"/api/v1/analytics/{neg}/game-theory", headers=AUTH)
    assert response.status_code == 403


def test_admin_analytics_open_when_auth_disabled(client):
    """No keys at all keeps local development open, as everywhere else."""
    assert client.get("/api/v1/analytics/aggregate").status_code == 200


def test_admin_keys_alone_enable_auth(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", [])
    monkeypatch.setattr(settings, "admin_api_keys", [ADMIN_KEY])
    assert auth.auth_enabled() is True


def test_warns_when_auth_on_but_no_admin_keys(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(settings, "api_keys", [KEY])
    monkeypatch.setattr(settings, "admin_api_keys", [])
    with caplog.at_level(logging.WARNING, logger=auth.__name__):
        auth.warn_if_unauthenticated("0.0.0.0")

    assert "ADMIN_API_KEYS is empty" in caplog.text
