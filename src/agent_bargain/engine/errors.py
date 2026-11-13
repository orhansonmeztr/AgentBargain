"""Domain errors raised while setting up a negotiation.

The engine does not know which transport it serves. It used to raise
``fastapi.HTTPException`` directly, which tied the core to FastAPI and made the
A2A server catch an HTTP type to report a protocol-level failure. Each error
carries the HTTP status it maps to, so the HTTP route can translate it in one
line and the A2A executor can surface the same ``detail`` text.
"""

from __future__ import annotations


class NegotiationSetupError(Exception):
    """A negotiation request that cannot be started as given."""

    #: HTTP status the HTTP surface answers with.
    status_code: int = 422

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ServiceNotRegisteredError(NegotiationSetupError):
    """The service has no listing, so there is no seller floor to negotiate against."""

    status_code = 404


class SellerMismatchError(NegotiationSetupError):
    """The request names a seller other than the listing's owner."""

    status_code = 409


class OpeningBelowFloorError(NegotiationSetupError):
    """The opening offer undercuts the seller's registered floor."""

    status_code = 422
