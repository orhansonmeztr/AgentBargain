"""Unit tests for the Discovery Engine."""

from unittest.mock import MagicMock

import pytest

from agent_bargain.engine.discovery import DiscoveryEngine, tokenize
from agent_bargain.storage.vector_store import VectorStoreManager


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStoreManager)
    # Mock search to return exact match IDs
    store.search_services.return_value = {"ids": [["service_1", "service_2"]]}
    store.get_metadatas.return_value = [{"title": "Web Dev"}, {"title": "API Design"}]
    return store


def test_discovery_index_and_search(mock_vector_store):
    engine = DiscoveryEngine(mock_vector_store)

    docs = ["Expert web development", "Backend API design"]
    metadatas = [{"title": "Web Dev"}, {"title": "API Design"}]
    ids = ["service_1", "service_2"]

    engine.index_services(ids, docs, metadatas)

    assert engine.bm25 is not None
    mock_vector_store.add_services.assert_called_once_with(ids, docs, metadatas)

    results = engine.search("web development")

    assert len(results) > 0
    assert results[0]["id"] in ids
    assert "rrf_score" in results[0]


def test_discovery_add_service_keeps_existing(mock_vector_store):
    engine = DiscoveryEngine(mock_vector_store)

    engine.index_services(["service_1"], ["Expert web development"], [{"title": "Web Dev"}])
    engine.add_service("service_2", "Backend API design", {"title": "API Design"})

    assert set(engine.service_ids) == {"service_1", "service_2"}
    assert engine.bm25 is not None


# -- Deferred BM25 rebuild ------------------------------------------


def test_add_service_is_visible_to_the_next_search(mock_vector_store):
    """Search must flush the deferred rebuild, not serve a stale index.

    ``add_service`` only marks the corpus dirty; if ``search`` forgot to flush,
    a newly listed service would be lexically unfindable forever.
    """
    mock_vector_store.get_metadatas.side_effect = lambda ids: [{"title": i} for i in ids]
    mock_vector_store.search_services.return_value = {"ids": [[]]}

    engine = DiscoveryEngine(mock_vector_store)
    engine.index_services(["service_1"], ["expert web development"], [{"title": "Web Dev"}])
    engine.add_service("service_2", "kubernetes cluster management", {"title": "K8s"})

    results = engine.search("kubernetes")

    ids = [r["id"] for r in results]
    assert "service_2" in ids, "newly added service missing from the index"
    assert ids[0] == "service_2", "the lexically matching service did not rank first"


def test_deferred_rebuild_amortises_writes(mock_vector_store):
    """N writes between searches cost one rebuild, not N."""
    engine = DiscoveryEngine(mock_vector_store)
    engine.index_services(["s0"], ["alpha service"], [{"title": "alpha"}])

    builds = 0
    original = engine._rebuild_bm25

    def _counting_rebuild():
        nonlocal builds
        builds += 1
        original()

    engine._rebuild_bm25 = _counting_rebuild

    for i in range(20):
        engine.add_service(f"s{i}", f"service number {i}", {"title": f"s{i}"})
    assert builds == 0, "add_service rebuilt the index eagerly"

    engine.search("service")
    assert builds == 1, "search did not flush the deferred rebuild exactly once"

    engine.search("service")
    assert builds == 1, "an unchanged corpus must not trigger another rebuild"


# -- Tokenization ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Expert Web Development", ["expert", "web", "development"]),
        ("API, design!", ["api", "design"]),
        ("kubernetes-based (RAG) endpoint", ["kubernetes", "based", "rag", "endpoint"]),
        ("v2 API: 50% off", ["v2", "api", "50", "off"]),
        ("", []),
    ],
)
def test_tokenize_strips_punctuation(text, expected):
    assert tokenize(text) == expected


def test_punctuation_does_not_hide_a_keyword(mock_vector_store):
    """A listing written "REST API," must be found by the query "api".

    With ``str.split()`` tokenization those were two unrelated terms and both
    documents scored 0.0, so ordering fell back to insertion order. The
    matching service is indexed *second* here precisely so that a stale stable
    sort cannot pass the test by accident.
    """
    mock_vector_store.get_metadatas.side_effect = lambda ids: [{"title": i} for i in ids]
    # No vector hits, so the ranking is decided by the lexical half alone.
    mock_vector_store.search_services.return_value = {"ids": [[]]}

    engine = DiscoveryEngine(mock_vector_store)
    engine.index_services(
        ["svc_other", "svc_api"],
        ["kubernetes cluster management", "REST API, integration"],
        [{"title": "K8s"}, {"title": "API"}],
    )

    ids = [r["id"] for r in engine.search("api")]

    assert ids[0] == "svc_api", f"punctuation hid the keyword; ranking was {ids}"


# -- Vector store factory -------------------------------------------


def test_create_vector_store_routes_to_chroma(monkeypatch, tmp_path):
    """The factory must build a ChromaDB-backed store when backend='chroma'."""
    import agent_bargain.storage.vector_store as vs_module

    sentinel = object()
    captured: dict[str, object] = {}

    def _fake_manager(persist_path=None):
        captured["persist_path"] = persist_path
        return sentinel

    monkeypatch.setattr(vs_module, "VectorStoreManager", _fake_manager)

    store = vs_module.create_vector_store("chroma", persist_path=tmp_path)
    assert store is sentinel
    assert captured["persist_path"] == tmp_path


def test_create_vector_store_defaults_to_settings_backend(monkeypatch, tmp_path):
    """Omitting the backend argument falls back to settings.vector_backend."""
    import agent_bargain.storage.vector_store as vs_module
    from agent_bargain.config import settings

    monkeypatch.setattr(settings, "vector_backend", "chroma")
    monkeypatch.setattr(vs_module, "VectorStoreManager", lambda persist_path=None: "store")

    assert vs_module.create_vector_store(persist_path=tmp_path) == "store"


def test_create_vector_store_rejects_unknown_backend():
    """Unsupported backends raise a clear ValueError instead of silently falling back."""
    import agent_bargain.storage.vector_store as vs_module

    with pytest.raises(ValueError, match="Unsupported vector backend"):
        vs_module.create_vector_store("milvus")  # type: ignore[arg-type]


def test_non_matching_documents_get_no_bm25_rank():
    """A document sharing no term with the query must not be boosted by RRF."""
    store = MagicMock()
    store.search_services.return_value = {"ids": [[]]}  # vector half silent
    store.get_metadatas.side_effect = lambda ids: [{"title": i} for i in ids]
    engine = DiscoveryEngine(store)
    engine.index_services(
        ["svc_gpu", "svc_web"],
        ["GPU training cluster", "Website hosting"],
        [{"title": "gpu"}, {"title": "web"}],
    )

    assert [r["id"] for r in engine.search("gpu", top_k=5)] == ["svc_gpu"]
