"""Unit tests for the negotiation engine and LangGraph state machine."""

from agent_bargain.engine.negotiation import (
    create_initial_state,
    create_negotiation_graph,
    create_strategy_graph,
)


def test_negotiation_graph_compilation():
    """Test that the negotiation graph compiles successfully."""
    app = create_negotiation_graph()

    assert app is not None
    nodes = app.get_graph().nodes
    assert "buyer" in nodes
    assert "seller" in nodes
    assert "accepted" in nodes
    assert "rejected" in nodes
    assert "timeout" in nodes
    assert "round_counter" in nodes


def test_graph_models_bargaining_only():
    """Post-trade evaluation must not be a node on the accepted path.

    While it was, the terminal status could not be persisted until a 15-40s LLM
    call finished, so a deal struck in ~50ms stayed "active" to every client for
    that long. Evaluation now happens after the outcome is durable.
    """
    for graph in (
        create_negotiation_graph(),
        create_strategy_graph("cooperative", "aggressive"),
    ):
        nodes = graph.get_graph().nodes
        assert "evaluator" not in nodes, "evaluator is back in the bargaining graph"

        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("accepted", "__end__") in edges, f"accepted does not end directly: {edges}"


def test_create_initial_state():
    """Test that the state factory produces valid defaults."""
    state = create_initial_state(
        service_id="srv_1",
        service_title="Test Service",
        buyer_id="buyer_1",
        seller_id="seller_1",
        max_budget=500.0,
        min_price=100.0,
        initial_price=300.0,
    )

    assert state["service_id"] == "srv_1"
    assert state["messages"] == []
    assert state["round"] == 0
    assert state["status"] == "active"
    assert state["last_action"] == ""


def test_initial_state_carries_no_evaluation_fields():
    """Evaluation is not bargaining state, so it must not ride along in the graph."""
    state = create_initial_state(
        service_id="srv_1",
        service_title="Test Service",
        buyer_id="buyer_1",
        seller_id="seller_1",
        max_budget=500.0,
        min_price=100.0,
        initial_price=300.0,
    )

    leftovers = [k for k in state if "evaluation" in k]
    assert not leftovers, f"evaluation fields still in the state: {leftovers}"
