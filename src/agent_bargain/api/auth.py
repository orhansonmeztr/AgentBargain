"""Shared-secret API key authentication for the /api/v1 surface.

Deliberately minimal: a static list of shared secrets from configuration,
compared in constant time. This is *authentication*, not authorisation — every
valid key has the same rights. There are no per-agent identities, roles or
scopes, so a key holder can act as any buyer or seller. Anything stronger needs
a real identity model and is out of scope here.

Policy:

* ``settings.api_keys`` and ``settings.admin_api_keys`` both empty ⇒
  authentication is **disabled**. This keeps local
  development frictionless, and ``create_app`` logs a warning (escalated when
  the server binds a non-loopback host).
* Otherwise every ``/api/v1`` router requires a key — including the A2A
  JSON-RPC endpoint, which starts negotiations and writes to the reputation
  ledger exactly like ``POST /negotiations/start``. Only the A2A Agent Card
  stays public: discovery is its entire purpose, and the card advertises the
  ``X-API-Key`` scheme so A2A clients know what to send.
* Binding a non-loopback host with authentication disabled is refused at
  startup (:func:`require_auth_when_exposed`) unless
  ``ALLOW_UNAUTHENTICATED=true`` opts in explicitly.

* Admin keys (``settings.admin_api_keys``) are a superset: valid wherever a
  regular key is, and the only keys accepted by :func:`verify_admin_api_key`.
  That guards the per-negotiation analytics, which necessarily reveal the
  seller's floor and the buyer's budget — the same secrets the listing and
  history responses strip. A regular key there gets 403, and with no admin key
  configured nobody gets in (fail closed). Negotiators' free-text reasoning
  follows the same rule (:func:`can_read_reasoning`): it is stripped from the
  transcript endpoint and the WebSocket stream for regular keys.

Keys are accepted as ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
Only WebSockets additionally accept ``?api_key=<key>``, because browsers cannot set
headers on ``new WebSocket()``; that puts the secret in the URL, so it should be
treated as a lesser transport and is documented as such.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
from typing import Annotated

from fastapi import HTTPException, Request, Security, WebSocket, status
from fastapi.security import APIKeyHeader

from agent_bargain.config import settings

logger = logging.getLogger(__name__)

#: Name of the header carrying the key.
API_KEY_HEADER = "X-API-Key"

_API_KEY_SCHEME = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="A valid API key is required. Send it as X-API-Key or Authorization: Bearer.",
    headers={"WWW-Authenticate": API_KEY_HEADER},
)


_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail=(
        "This route reveals negotiators' private limits and requires an admin key (ADMIN_API_KEYS)."
    ),
)


def auth_enabled() -> bool:
    """True when at least one regular or admin API key is configured."""
    return bool(settings.api_keys or settings.admin_api_keys)


def _matches(candidate: str | None, keys: list[str] | None = None) -> bool:
    """Constant-time comparison against every configured key.

    ``==`` on a secret leaks its length and matching prefix through timing.
    Every key is always compared so the cost does not depend on which one (if
    any) matched — hence a list, not a generator: ``any()`` over a generator
    stops at the first match. ``keys`` defaults to regular plus admin keys.
    """
    if not candidate:
        return False
    if keys is None:
        keys = _all_keys()
    supplied = candidate.encode()
    # Iterated rather than short-circuited on purpose.
    return any([hmac.compare_digest(supplied, k.encode()) for k in keys])


def _all_keys() -> list[str]:
    return [*settings.api_keys, *settings.admin_api_keys]


def _presented_keys(request_headers, query_key: str | None, header_key: str | None):
    """Every credential the client presented, in the transports we accept."""
    return (
        header_key,
        _bearer_token(request_headers.get("Authorization")),
        query_key,
    )


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def headers_authenticate(headers) -> bool:
    """True when auth is off, or ``headers`` carry a valid regular/admin key.

    For callers outside FastAPI's dependency system (the A2A middleware).
    Headers only, like :func:`verify_api_key`.
    """
    if not auth_enabled():
        return True
    keys = _all_keys()
    presented = _presented_keys(headers, None, headers.get(API_KEY_HEADER))
    return any(_matches(c, keys) for c in presented)


async def verify_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Security(_API_KEY_SCHEME)] = None,
) -> None:
    """FastAPI dependency: reject the request unless a configured key is presented.

    Headers only. ``?api_key=`` is deliberately *not* read here: it exists for
    the WebSocket, whose browser API cannot set headers, and accepting it on
    plain HTTP put the secret into access logs and browser history for no
    reason.

    The scheme must be wrapped in ``Security()``: passed as a bare default,
    FastAPI classifies it as a *query* parameter and every protected route
    answers 422 before the handler runs. ``Authorization`` is read off the
    request directly so it does not pollute the OpenAPI schema either.
    """
    if not auth_enabled():
        return

    keys = _all_keys()
    if any(_matches(c, keys) for c in _presented_keys(request.headers, None, x_api_key)):
        return

    raise _UNAUTHENTICATED


async def verify_admin_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Security(_API_KEY_SCHEME)] = None,
) -> None:
    """FastAPI dependency: require an *admin* key (401 none valid, 403 regular only).

    Open when authentication is disabled, like every other route — local
    development keeps working without configuring keys.
    """
    if not auth_enabled():
        return

    presented = _presented_keys(request.headers, None, x_api_key)
    if any(_matches(c, settings.admin_api_keys) for c in presented):
        return
    if any(_matches(c, settings.api_keys) for c in presented):
        raise _FORBIDDEN
    raise _UNAUTHENTICATED


def verify_websocket_api_key(websocket: WebSocket, api_key: str | None = None) -> bool:
    """Return True when the WebSocket handshake may be accepted. Kept separate from the HTTP
    dependency because a WebSocket cannot answer with an HTTP 401 once the
    handshake has been accepted — the handshake must be refused instead.
    """
    if not auth_enabled():
        return True
    keys = _all_keys()
    presented = _presented_keys(websocket.headers, api_key, websocket.headers.get(API_KEY_HEADER))
    return any(_matches(c, keys) for c in presented)


def can_read_reasoning(headers, query_key: str | None = None) -> bool:
    """True when the caller may see negotiators' free-text reasoning.

    Reasoning routinely states the party's own limit ("my budget is $300"), and
    a regular key can act as any party, so it is admin-only — the same rule as
    the game-theory analytics. Open when authentication is disabled, like
    every other route. ``query_key`` is the WebSocket's ``?api_key=``.
    """
    if not auth_enabled():
        return True
    presented = _presented_keys(headers, query_key, headers.get(API_KEY_HEADER))
    return any(_matches(c, settings.admin_api_keys) for c in presented)


def warn_if_unauthenticated(host: str) -> None:
    """Log the security posture at startup. Called once from the app factory."""
    if auth_enabled():
        logger.info(
            "API key authentication enabled (%d key(s), %d admin key(s) configured).",
            len(settings.api_keys),
            len(settings.admin_api_keys),
        )
        if not settings.admin_api_keys:
            logger.warning(
                "ADMIN_API_KEYS is empty — the per-negotiation analytics routes "
                "(game-theory, aggregate) will answer 403 to every caller."
            )
        return

    if is_loopback(host):
        logger.warning(
            "API_KEYS is empty — /api/v1 is UNAUTHENTICATED. Acceptable while "
            "bound to %s; set API_KEYS before exposing the port.",
            host,
        )
    else:
        logger.warning(
            "API_KEYS is empty and SERVER_HOST=%s (allowed by "
            "ALLOW_UNAUTHENTICATED): anyone able to reach this port can start "
            "negotiations and write to the reputation ledger. "
            "Set API_KEYS, e.g. API_KEYS='[\"change-me\"]'.",
            host,
        )


def is_loopback(host: str) -> bool:
    """True for addresses only this machine can reach."""
    if host.strip("[]").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def require_auth_when_exposed(host: str) -> None:
    """Refuse to serve an unauthenticated API on a reachable interface.

    A warning alone was the only guard, and the default host used to be
    ``0.0.0.0``: a fresh install exposed the whole write surface to the
    network. Opting out must now be explicit (``ALLOW_UNAUTHENTICATED=true``).
    """
    if auth_enabled() or is_loopback(host) or settings.allow_unauthenticated:
        return
    raise RuntimeError(
        f"Refusing to start: SERVER_HOST={host} is reachable from other machines "
        "but API_KEYS and ADMIN_API_KEYS are empty, so /api/v1 would be open to "
        "anyone. Set API_KEYS (e.g. API_KEYS='[\"change-me\"]'), bind "
        "SERVER_HOST=127.0.0.1, or set ALLOW_UNAUTHENTICATED=true to accept the risk."
    )
