"""Vector store abstraction for hybrid search.

Defines the ``VectorStore`` protocol so alternative backends (ChromaDB,
Milvus, FAISS, ...) can be swapped in without touching the discovery engine.
The bundled ``VectorStoreManager`` is a ChromaDB implementation.

Backend selection is controlled by ``VECTOR_BACKEND`` in the environment
(default: ``"chroma"``).  Additional backends can be added by implementing
the ``VectorStore`` protocol and registering them in ``create_vector_store()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from agent_bargain.config import settings


class VectorStore(Protocol):
    """Minimal interface the discovery engine depends on."""

    def add_services(
        self,
        service_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update services in the store."""
        ...

    def search_services(self, query_text: str, n_results: int = 5) -> dict[str, Any]:
        """Return ranked semantic search results."""
        ...

    def get_metadatas(self, ids: list[str]) -> list[dict[str, Any] | None]:
        """Return metadata for the given IDs, preserving order."""
        ...


class VectorStoreManager:
    """Manages ChromaDB vector store operations.

    Accepts an optional path override for testability. Falls back to the
    global settings when no path is provided.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        resolved_path = persist_path or settings.chroma_persist_full_path
        self.client = chromadb.PersistentClient(
            path=str(resolved_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name="services")

    def add_services(
        self,
        service_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or update service listings in the vector store."""
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=service_ids,
        )

    def search_services(self, query_text: str, n_results: int = 5) -> dict[str, Any]:
        """Perform a semantic search for services."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

    def get_metadatas(self, ids: list[str]) -> list[dict[str, Any] | None]:
        """Return metadata for the given IDs, preserving order."""
        if not ids:
            return []
        result = self.collection.get(ids=ids)
        metadatas = (result or {}).get("metadatas") or []
        # Map back by id so ordering is preserved and missing ids become None.
        by_id = {
            sid: (metadatas[i] if i < len(metadatas) else None)
            for i, sid in enumerate((result or {}).get("ids", []))
        }
        return [by_id.get(sid) for sid in ids]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Supported backend identifiers.
VectorBackend = Literal["chroma"]


def create_vector_store(
    backend: VectorBackend | None = None,
    persist_path: Path | None = None,
) -> VectorStore:
    """Instantiate and return a ``VectorStore`` implementation.

    Args:
        backend:      Which backend to use.  Reads ``VECTOR_BACKEND`` from the
                      environment when *None*; defaults to ``"chroma"``.
        persist_path: Override for the persistence directory (useful in tests).

    Returns:
        A ``VectorStore``-compatible object.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    resolved_backend: str = backend or getattr(settings, "vector_backend", "chroma")

    if resolved_backend == "chroma":
        return VectorStoreManager(persist_path=persist_path)

    raise ValueError(
        f"Unsupported vector backend: {resolved_backend!r}. "
        f"Supported: {list(VectorBackend.__args__)}"  # type: ignore[attr-defined]
    )
