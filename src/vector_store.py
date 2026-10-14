"""Local semantic index (Chroma + on-device MiniLM embeddings).

The vector index is optional by design. It only re-ranks chunks that the structured
database has already restricted to one mine, so if the embedding model cannot be loaded
(for example on an air-gapped machine before the model file is provisioned) retrieval
degrades to keyword ranking and the application keeps working.

No text leaves the machine: embeddings are computed locally with onnxruntime.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

COLLECTION = "sanket_chunks"


class VectorStoreUnavailable(RuntimeError):
    pass


class VectorStore:
    def __init__(self, directory: Path, backend: str = "onnx-minilm"):
        self.directory = Path(directory)
        self.backend = backend
        self._collection = None

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def _open(self):
        if self._collection is not None:
            return self._collection
        if not self.enabled:
            raise VectorStoreUnavailable("Vector index disabled (SANKET_EMBEDDING_BACKEND=none).")
        if self.backend != "onnx-minilm":
            raise VectorStoreUnavailable(f"Unknown embedding backend {self.backend!r}.")
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            self.directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.directory), settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = client.get_or_create_collection(
                COLLECTION, embedding_function=ONNXMiniLM_L6_V2(), metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:  # import, model download and disk errors all end up here
            raise VectorStoreUnavailable(f"Vector index unavailable: {exc}") from exc
        return self._collection

    def count(self) -> int:
        try:
            return self._open().count()
        except VectorStoreUnavailable:
            return 0

    def sync(self, chunks: list[dict], batch_size: int = 128) -> dict[str, int]:
        """Make the collection mirror ``chunks``: upsert by stable id, delete stale ids."""
        collection = self._open()
        try:
            existing = set(collection.get(include=[])["ids"])
            wanted = {c["chunk_id"] for c in chunks}
            stale = sorted(existing - wanted)
            for start in range(0, len(stale), 500):
                collection.delete(ids=stale[start : start + 500])
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                collection.upsert(
                    ids=[c["chunk_id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[
                        {"mine_id": c["mine_id"], "incident_id": c["incident_id"],
                         "source_id": c["source_id"], "chunk_type": c["chunk_type"]}
                        for c in batch
                    ],
                )
        except Exception as exc:
            raise VectorStoreUnavailable(f"Vector indexing failed: {exc}") from exc
        return {"indexed": len(chunks), "deleted_stale": len(stale), "total": collection.count()}

    def query(self, mine_id: str, text: str, limit: int = 10) -> list[dict]:
        """Semantic ranking strictly inside one mine (exact metadata filter)."""
        collection = self._open()
        try:
            res = collection.query(
                query_texts=[text], n_results=limit, where={"mine_id": mine_id},
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreUnavailable(f"Vector query failed: {exc}") from exc
        return [
            {"chunk_id": cid, "distance": dist, **meta}
            for cid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
        ]
