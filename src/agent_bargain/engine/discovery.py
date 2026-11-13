"""Hybrid discovery engine combining BM25 and Vector Search."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from rank_bm25 import BM25L

from agent_bargain.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

#: Reciprocal Rank Fusion damping constant. Larger values flatten the
#: difference between ranks; 60 is the conventional default. Both rankers are
#: weighted equally — there is no tuning knob for that yet.
RRF_K = 60

#: Word tokens: letters, digits and underscores. Punctuation is a separator
#: rather than part of a term.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for BM25.

    ``text.lower().split()`` treats punctuation as part of a term, so "API,"
    and "api" indexed as two unrelated words and a trailing period silently
    removed a keyword from the lexical half of the hybrid search.
    """
    return _TOKEN_RE.findall(text.lower())


def service_document(service: dict[str, Any]) -> str:
    """Build the searchable text for a service listing dict."""
    parts = [
        str(service.get("title", "")),
        str(service.get("description", "")),
        str(service.get("category_name", "")),
    ]
    return " ".join(p for p in parts if p).strip() or "Unnamed service"


def service_metadata(service: dict[str, Any]) -> dict[str, Any]:
    """Build the metadata stored alongside a service in the index.

    Takes a service row (``category_name`` flattened, as the database returns
    it). Never includes ``floor_price``: index metadata is returned verbatim by
    every search.
    """
    return {
        "title": service.get("title", ""),
        "category": service.get("category_name", ""),
        "seller_id": service.get("seller_id", ""),
    }


def corpus_from_rows(
    services: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """``(ids, documents, metadatas)`` for a list of service rows.

    The argument tuple of :meth:`DiscoveryEngine.index_services` and
    :meth:`DiscoveryEngine.load_corpus`, so the API server and the MCP server
    seed their indexes from one definition.
    """
    return (
        [s["service_id"] for s in services],
        [service_document(s) for s in services],
        [service_metadata(s) for s in services],
    )


class DiscoveryEngine:
    """Hybrid search combining BM25 lexical and ChromaDB vector semantic search.

    Keeps an in-memory mirror of the indexed corpus so services can be added
    incrementally without losing previously indexed entries.

    Both ``search`` and the mutators perform blocking I/O (ChromaDB) and CPU
    work (BM25 scoring), so callers on the event loop must dispatch through
    ``run_in_threadpool``. The internal lock makes that safe: a search and a
    concurrent ``add_service`` would otherwise observe a half-updated corpus.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.bm25: BM25L | None = None
        self.service_ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._bm25_stale = False
        # Reentrant: search() holds it while calling _rebuild_bm25().
        self._lock = threading.RLock()

    def index_services(
        self,
        service_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Replace the full index in both ChromaDB and BM25.

        The index is built eagerly here: this is the bulk seed path, so paying
        for one build is cheaper than deferring it into the first search.
        """
        if not service_ids:
            return

        with self._lock:
            # 1. Vector Store
            self.vector_store.add_services(service_ids, documents, metadatas)

            # 2. Lexical (BM25)
            self.service_ids = list(service_ids)
            self._documents = list(documents)
            self._metadatas = list(metadatas)
            self._rebuild_bm25()

    def load_corpus(
        self,
        service_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Replace the in-memory (BM25) corpus *without* writing the vector store.

        For read-only consumers such as the MCP server: the vector store is
        persisted on disk and owned by the API server, so re-upserting it from a
        second process would be a concurrent write to the same Chroma directory.
        Vector search then reflects whatever the owner last indexed.
        """
        with self._lock:
            self.service_ids = list(service_ids)
            self._documents = list(documents)
            self._metadatas = list(metadatas)
            self._rebuild_bm25()

    def add_service(self, service_id: str, document: str, metadata: dict[str, Any]) -> None:
        """Insert or update a single service in the index.

        The BM25 rebuild is deferred to the next search. ``BM25L`` has no
        incremental update, so rebuilding here would make indexing O(corpus)
        per negotiation and O(corpus²) across a burst of them; deferring
        amortises any number of writes into a single rebuild.
        """
        with self._lock:
            if service_id in self.service_ids:
                idx = self.service_ids.index(service_id)
                self._documents[idx] = document
                self._metadatas[idx] = metadata
            else:
                self.service_ids.append(service_id)
                self._documents.append(document)
                self._metadatas.append(metadata)

            self.vector_store.add_services([service_id], [document], [metadata])
            self._bm25_stale = True

    def _rebuild_bm25(self) -> None:
        # BM25L rather than BM25Okapi: Okapi's IDF is log((N-df+0.5)/(df+0.5)),
        # which is exactly 0 when a term appears in half the corpus — so in a
        # two-service marketplace every lexical score came out 0 and the RRF
        # ranking silently degraded to insertion order. Okapi also goes
        # *negative* on a single-document corpus. BM2L keeps a hard 0 for
        # non-matching documents (so they contribute nothing to RRF) without
        # collapsing on small corpora. BM25Plus was rejected: it gives every
        # document a positive floor, which dilutes the fusion signal.
        tokenized_corpus = [tokenize(doc) for doc in self._documents]
        self.bm25 = BM25L(tokenized_corpus) if tokenized_corpus else None
        self._bm25_stale = False

    def _ensure_bm25(self) -> None:
        if self._bm25_stale:
            self._rebuild_bm25()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Perform hybrid search using Reciprocal Rank Fusion (RRF)."""
        with self._lock:
            if not self.service_ids:
                return []
            self._ensure_bm25()
            if self.bm25 is None:
                return []

            # 1. Vector Search
            vector_results = self.vector_store.search_services(query, n_results=top_k * 2)

            # 2. Lexical Search
            bm25_scores = self.bm25.get_scores(tokenize(query))

            # Rank BM25 results. Zero-score documents share no term with the
            # query; ranking them anyway handed non-matches an RRF boost purely
            # by corpus position.
            bm25_ranked = sorted(
                (
                    (sid, score)
                    for sid, score in zip(self.service_ids, bm25_scores, strict=False)
                    if score > 0
                ),
                key=lambda x: x[1],
                reverse=True,
            )[: top_k * 2]

            # 3. Reciprocal Rank Fusion (RRF)
            rrf_scores: dict[str, float] = {}

            # Add vector ranks
            if vector_results and "ids" in vector_results and vector_results["ids"]:
                v_ids = vector_results["ids"][0]
                for rank, s_id in enumerate(v_ids):
                    rrf_scores[s_id] = rrf_scores.get(s_id, 0.0) + (1.0 / (RRF_K + rank + 1))

            # Add BM25 ranks
            for rank, (s_id, _) in enumerate(bm25_ranked):
                rrf_scores[s_id] = rrf_scores.get(s_id, 0.0) + (1.0 / (RRF_K + rank + 1))

            # Sort combined scores
            final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Fetch metadata from the vector store, falling back to the in-memory
            # mirror: a read-only corpus (see load_corpus) can know services the
            # vector store has not indexed yet, and BM25 may rank them.
            ranked_ids = [r[0] for r in final_ranked]
            local = dict(zip(self.service_ids, self._metadatas, strict=False))
            metadatas = [
                stored if stored is not None else local.get(sid)
                for sid, stored in zip(
                    ranked_ids, self.vector_store.get_metadatas(ranked_ids), strict=False
                )
            ]
            final_results: list[dict[str, Any]] = []
            for s_id, metadata in zip(ranked_ids, metadatas, strict=False):
                if metadata is not None:
                    final_results.append(
                        {
                            "id": s_id,
                            "metadata": metadata,
                            "rrf_score": rrf_scores.get(s_id, 0.0),
                        }
                    )

            return sorted(final_results, key=lambda x: x["rrf_score"], reverse=True)
