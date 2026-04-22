from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim: int) -> None:
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        index = faiss.read_index(str(path))
        store = cls(index.d)
        store.index = index
        return store
