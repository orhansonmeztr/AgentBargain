"""Tests for Game Theory Analytics (Nash, Pareto, ZOPA, BATNA, Concession)."""

import json

from agent_bargain.analytics.game_theory import GameTheoryAnalyzer


def make_analyzer():
    return GameTheoryAnalyzer()


# -- Successful negotiation -----------------------------------------


def test_accepted_negotiation_full_analysis():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_test1",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=5,
        max_rounds=10,
    )
    assert r.outcome == "accepted"
    assert r.final_price == 300.0

    # ZOPA
    assert r.zopa.exists is True
    assert r.zopa.size == 400.0
    assert r.zopa.lower_bound == 100.0
    assert r.zopa.upper_bound == 500.0

    # Nash: midpoint of ZOPA = (500+100)/2 = 300
    assert r.nash.nash_price == 300.0
    assert r.nash.distance == 0.0  # Perfect Nash deal!
    assert r.nash.distance_pct == 0.0

    # Pareto: 100 <= 300 <= 500 → efficient
    assert r.pareto.is_efficient is True
    assert r.pareto.buyer_surplus == 200.0
    assert r.pareto.seller_surplus == 200.0
    assert r.pareto.egalitarian_score == 1.0  # Equal split → perfect

    # BATNA: both parties above BATNA
    assert r.batna.buyer_batna_distance == 200.0
    assert r.batna.seller_batna_distance == 200.0
    assert r.batna.both_above_batna is True

    # Efficiency & Fairness: perfect Nash deal
    assert r.efficiency_score > 0.5
    assert r.fairness_score > 0.8


def test_buyer_favored_deal():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_buyer_favored",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=150.0,  # Far below Nash=300
        outcome="accepted",
        rounds_used=3,
        max_rounds=10,
    )
    assert r.nash.buyer_favored is True
    assert r.pareto.buyer_surplus == 350.0
    assert r.pareto.seller_surplus == 50.0
    assert r.pareto.surplus_ratio is not None
    assert r.pareto.surplus_ratio > 1.0  # buyer surplus > seller
    assert r.pareto.egalitarian_score < 0.5  # Very unequal


def test_seller_favored_deal():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_seller_favored",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=450.0,  # Far above Nash=300
        outcome="accepted",
        rounds_used=8,
        max_rounds=10,
    )
    assert r.nash.buyer_favored is False
    assert r.pareto.seller_surplus == 350.0
    assert r.pareto.buyer_surplus == 50.0


# -- Rejected/Timeout scenarios -------------------------------------


def test_rejected_negotiation():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_rejected",
        buyer_max=200.0,
        seller_min=300.0,  # No ZOPA!
        initial_price=350.0,
        final_price=None,
        outcome="rejected",
        rounds_used=4,
        max_rounds=10,
    )
    assert r.outcome == "rejected"
    assert r.final_price is None
    assert r.zopa.exists is False
    assert r.zopa.size < 0
    assert r.efficiency_score == 0.0


def test_timeout_negotiation():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_timeout",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=None,
        outcome="timeout",
        rounds_used=10,
        max_rounds=10,
    )
    assert r.outcome == "timeout"
    assert r.efficiency_score == 0.0
    assert r.pareto.is_efficient is False


# -- Edge cases -----------------------------------------------------


def test_equal_zopa_bounds():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_tight",
        buyer_max=100.0,
        seller_min=100.0,
        initial_price=100.0,
        final_price=100.0,
        outcome="accepted",
        rounds_used=1,
        max_rounds=10,
    )
    assert r.zopa.exists is True
    assert r.zopa.size == 0.0
    assert r.pareto.total_surplus == 0.0


def test_concession_analysis_with_trajectory():
    a = make_analyzer()
    trajectory = [400.0, 380.0, 350.0, 320.0, 300.0]
    r = a.analyze(
        negotiation_id="neg_trajectory",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=5,
        max_rounds=10,
        price_trajectory=trajectory,
    )
    assert r.concession.total_rounds == 5
    assert r.concession.first_offer == 400.0
    assert r.concession.final_price == 300.0
    assert r.concession.price_trajectory == trajectory
    assert r.concession.seller_total_concession == 100.0  # 400→300


def test_to_dict_serialization():
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_serialize",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=280.0,
        outcome="accepted",
        rounds_used=4,
        max_rounds=10,
    )
    d = r.to_dict()
    assert isinstance(d, dict)
    assert "zopa" in d
    assert "nash" in d
    assert "pareto" in d
    assert "batna" in d
    assert "concession" in d
    assert isinstance(d["zopa"]["size"], float)
    assert isinstance(d["nash"]["nash_price"], float)
    assert isinstance(d["pareto"]["egalitarian_score"], float)


def test_surplus_ratio_none_when_seller_at_min():
    """Seller accepting at their exact minimum produces surplus_ratio=None (not inf).

    This ensures GET /api/v1/analytics/.../game-theory never raises
    ValueError: Out of range float values are not JSON compliant.
    """
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_tight_margin",
        buyer_max=250.0,
        seller_min=200.0,
        initial_price=240.0,
        final_price=200.0,  # seller accepted at their exact floor
        outcome="accepted",
        rounds_used=2,
        max_rounds=8,
    )
    assert r.pareto.seller_surplus == 0.0
    assert r.pareto.surplus_ratio is None

    # Must not raise with allow_nan=False (simulates FastAPI JSONResponse behaviour)
    payload = json.dumps(r.to_dict(), allow_nan=False)
    d = json.loads(payload)
    assert d["pareto"]["surplus_ratio"] is None


def test_to_dict_allow_nan_safe_for_normal_deal():
    """Normal deals (both parties with surplus) must also serialize cleanly."""
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_normal",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=5,
        max_rounds=10,
    )
    payload = json.dumps(r.to_dict(), allow_nan=False)
    d = json.loads(payload)
    assert isinstance(d["pareto"]["surplus_ratio"], float)
    assert d["pareto"]["surplus_ratio"] == 1.0  # equal surplus → ratio 1.0


# -- Concession attribution ----------------------------------------


def test_concessions_attributed_per_actor():
    """With actor labels, concessions are measured against each party's own offers."""
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_attributed",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=2,
        max_rounds=10,
        price_trajectory=[400.0, 250.0, 380.0, 300.0],
        trajectory_actors=["initial", "buyer", "seller", "buyer"],
    )
    c = r.concession
    assert c.attributed is True
    # Buyer moved 250 → 300 (paying more = concession)
    assert c.buyer_total_concession == 50.0
    # Seller has only one own offer (380); no concession measurable yet
    assert c.seller_total_concession == 0.0
    assert c.buyer_retraction == 0.0
    assert c.seller_retraction == 0.0


def test_concessions_track_retractions_separately():
    """A move against a party's own interest is a retraction, not a concession."""
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_retraction",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=430.0,
        outcome="accepted",
        rounds_used=2,
        max_rounds=10,
        price_trajectory=[400.0, 250.0, 420.0, 240.0, 430.0],
        trajectory_actors=["initial", "buyer", "seller", "buyer", "seller"],
    )
    c = r.concession
    assert c.attributed is True
    assert c.buyer_retraction == 10.0  # 250 → 240 (buyer hardening)
    assert c.seller_retraction == 10.0  # 420 → 430 (seller hardening)
    assert c.buyer_total_concession == 0.0
    assert c.seller_total_concession == 0.0


def test_concessions_fall_back_without_actor_labels():
    """Without matching actor labels the analyzer keeps the net-movement heuristic."""
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_no_actors",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=5,
        max_rounds=10,
        price_trajectory=[400.0, 380.0, 300.0],
    )
    assert r.concession.attributed is False
    assert r.concession.seller_total_concession == 100.0
    assert r.concession.buyer_total_concession == 0.0

    # Mismatched label count also disables attribution.
    r2 = a.analyze(
        negotiation_id="neg_bad_actors",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=5,
        max_rounds=10,
        price_trajectory=[400.0, 300.0],
        trajectory_actors=["buyer"],
    )
    assert r2.concession.attributed is False


def test_concession_attribution_is_serialized():
    """to_dict exposes attribution fields and stays JSON-safe."""
    a = make_analyzer()
    r = a.analyze(
        negotiation_id="neg_serialize_attr",
        buyer_max=500.0,
        seller_min=100.0,
        initial_price=400.0,
        final_price=300.0,
        outcome="accepted",
        rounds_used=2,
        max_rounds=10,
        price_trajectory=[400.0, 250.0, 380.0, 300.0],
        trajectory_actors=["initial", "buyer", "seller", "buyer"],
    )
    d = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    assert d["concession"]["attributed"] is True
    assert d["concession"]["buyer_total_concession"] == 50.0
    assert d["concession"]["buyer_retraction"] == 0.0
    assert d["concession"]["seller_retraction"] == 0.0
