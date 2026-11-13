"""The seller's floor price must come from the seller, not the counterparty.

Before this, ``POST /negotiations/start`` accepted ``min_price`` in its body —
so the buyer declared the seller's walk-away point and could send ``min_price=0``
to win every negotiation. The game-theory analytics then measured a game the
seller was never actually playing.

The floor now lives on the registered service listing, and the listing is
created by ``POST /api/v1/marketplace/services``.
"""

from __future__ import annotations

from tests.conftest import DEFAULT_SERVICE, register_service, start_payload

# -- registration ---------------------------------------------------


def test_register_service_creates_a_searchable_listing(client):
    response = client.post("/api/v1/marketplace/services", json=dict(DEFAULT_SERVICE))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "registered"
    assert body["service"]["service_id"] == DEFAULT_SERVICE["service_id"]

    # Immediately discoverable through hybrid search.
    found = client.post(
        "/api/v1/marketplace/search",
        json={"query": DEFAULT_SERVICE["title"], "top_k": 5},
    ).json()["results"]
    assert DEFAULT_SERVICE["service_id"] in {r["id"] for r in found}


def test_register_service_is_idempotent(client):
    """Re-registering updates the listing rather than failing or duplicating."""
    register_service(client)
    response = client.post(
        "/api/v1/marketplace/services",
        json={**DEFAULT_SERVICE, "base_price": 999.0, "floor_price": 250.0},
    )

    assert response.status_code == 201
    services = client.get("/api/v1/marketplace/services").json()["services"]
    assert len(services) == 1
    assert services[0]["base_price"] == 999.0


def test_register_service_validates_inputs(client):
    assert (
        client.post(
            "/api/v1/marketplace/services",
            json={**DEFAULT_SERVICE, "floor_price": -1.0},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/marketplace/services",
            json={**DEFAULT_SERVICE, "service_id": ""},
        ).status_code
        == 422
    )


# -- floor price confidentiality ----------------------------------


def test_floor_price_is_never_leaked_to_buyers(client):
    """The whole point of a floor is that the other side cannot read it."""
    register_service(client, floor_price=275.0)

    single = client.get(f"/api/v1/marketplace/services/{DEFAULT_SERVICE['service_id']}")
    assert single.status_code == 200
    assert "floor_price" not in single.json()

    listing = client.get("/api/v1/marketplace/services")
    for row in listing.json()["services"]:
        assert "floor_price" not in row

    register = client.post("/api/v1/marketplace/services", json=dict(DEFAULT_SERVICE))
    assert "floor_price" not in register.json()["service"]

    # Discovery metadata must not carry it either.
    results = client.post(
        "/api/v1/marketplace/search", json={"query": DEFAULT_SERVICE["title"], "top_k": 5}
    ).json()["results"]
    for result in results:
        assert "floor_price" not in result["metadata"]
        assert 275.0 not in result["metadata"].values()


def test_unknown_service_is_404(client):
    assert client.get("/api/v1/marketplace/services/nope").status_code == 404


# -- negotiation start ----------------------------------------------


def test_start_requires_a_registered_service(client):
    """/start must not silently invent a listing with a buyer-chosen floor."""
    response = client.post("/api/v1/negotiations/start", json=start_payload())

    assert response.status_code == 404
    assert "not registered" in response.json()["detail"]


def test_start_rejects_a_buyer_supplied_floor(client):
    """``min_price`` is gone from the request schema entirely."""
    register_service(client)

    payload = start_payload()
    payload["min_price"] = 0.0  # the old exploit
    response = client.post("/api/v1/negotiations/start", json=payload)

    assert response.status_code == 200, response.text
    negotiation_id = response.json()["negotiation_id"]

    # The extra field is ignored; the registered floor still governs the deal.
    analysis = client.get(f"/api/v1/analytics/{negotiation_id}/game-theory").json()
    assert analysis["zopa"]["lower_bound"] == DEFAULT_SERVICE["floor_price"]


def test_start_rejects_the_wrong_seller(client):
    register_service(client, seller_id="real_seller")

    response = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(seller_id="some_other_agent"),
    )

    assert response.status_code == 409
    assert "real_seller" in response.json()["detail"]


def test_start_rejects_an_opening_offer_below_the_floor(client):
    register_service(client, floor_price=200.0, base_price=400.0)

    response = client.post("/api/v1/negotiations/start", json=start_payload(initial_price=150.0))

    assert response.status_code == 422
    assert "floor" in response.json()["detail"].lower()


def test_opening_offer_defaults_to_the_listed_base_price(client):
    register_service(client, base_price=420.0, floor_price=100.0)

    payload = start_payload(
        engine="strategy", buyer_strategy="cooperative", seller_strategy="cooperative"
    )
    del payload["initial_price"]
    response = client.post("/api/v1/negotiations/start", json=payload)
    assert response.status_code == 200, response.text

    status = client.get(f"/api/v1/negotiations/{response.json()['negotiation_id']}/status").json()
    analysis = client.get(f"/api/v1/analytics/{status['negotiation_id']}/game-theory").json()

    assert analysis["concession"]["first_offer"] == 420.0


def test_registered_floor_governs_the_negotiation(client):
    """End to end: the ZOPA the analytics reports is built from the listing."""
    register_service(client, floor_price=180.0, base_price=400.0)

    response = client.post(
        "/api/v1/negotiations/start",
        json=start_payload(
            max_budget=500.0,
            engine="strategy",
            buyer_strategy="cooperative",
            seller_strategy="cooperative",
        ),
    )
    assert response.status_code == 200, response.text
    negotiation_id = response.json()["negotiation_id"]

    status = client.get(f"/api/v1/negotiations/{negotiation_id}/status").json()
    assert status["status"] == "accepted"

    analysis = client.get(f"/api/v1/analytics/{negotiation_id}/game-theory").json()
    assert analysis["zopa"]["lower_bound"] == 180.0
    assert analysis["zopa"]["upper_bound"] == 500.0
    # The seller's guardrail refuses to settle below its own floor.
    assert analysis["final_price"] >= 180.0


def test_another_seller_cannot_take_over_a_listing(client):
    """Re-registering someone else's service_id must not transfer it."""
    register_service(client)

    response = client.post(
        "/api/v1/marketplace/services",
        json={**DEFAULT_SERVICE, "seller_id": "intruder", "floor_price": 0.0},
    )
    assert response.status_code == 409

    listing = client.get(f"/api/v1/marketplace/services/{DEFAULT_SERVICE['service_id']}").json()
    assert listing["seller_id"] == DEFAULT_SERVICE["seller_id"]


def test_owner_can_still_update_its_listing(client):
    register_service(client)
    register_service(client, base_price=450.0)
    listing = client.get(f"/api/v1/marketplace/services/{DEFAULT_SERVICE['service_id']}").json()
    assert listing["base_price"] == 450.0
