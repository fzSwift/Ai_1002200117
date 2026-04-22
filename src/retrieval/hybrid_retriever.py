from __future__ import annotations

from typing import Any

import numpy as np

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.scoring import classify_query, compute_domain_bonus, normalize_scores
from src.retrieval.vector_store import VectorStore


def _token_overlap_ratio(query: str, text: str) -> float:
    q = set(query.lower().split())
    if not q:
        return 0.0
    t = set(text.lower().split())
    return len(q & t) / len(q)


class HybridRetriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]
        self.embedder = Embedder()
        self.embeddings = self.embedder.encode(self.texts)
        self.vector_store = VectorStore(self.embeddings.shape[1])
        self.vector_store.add(self.embeddings)
        self.bm25 = BM25Retriever(self.texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        initial_k: int = 8,
        *,
        classification_query: str | None = None,
    ) -> dict[str, Any]:
        """`query` is used for embedding/BM25 (may be rewritten). `classification_query` defaults to `query` for election/budget routing."""
        qc = classification_query if classification_query is not None else query
        query_type = classify_query(qc)
        query_embedding = self.embedder.encode([query])[0]

        vector_scores, vector_indices = self.vector_store.search(query_embedding, top_k=initial_k)
        bm25_scores, bm25_indices = self.bm25.search(query, top_k=initial_k)

        vector_scores_list = [float(score) for score in vector_scores.tolist()]
        bm25_scores_list = [float(score) for score in bm25_scores]

        norm_vector_scores = normalize_scores(vector_scores_list)
        norm_bm25_scores = normalize_scores(bm25_scores_list)

        candidates: dict[int, dict] = {}

        for idx, score, norm_score in zip(vector_indices.tolist(), vector_scores_list, norm_vector_scores):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            candidates[idx] = {
                **chunk,
                "vector_score": score,
                "vector_score_norm": norm_score,
                "bm25_score": 0.0,
                "bm25_score_norm": 0.0,
            }

        for idx, score, norm_score in zip(bm25_indices, bm25_scores_list, norm_bm25_scores):
            chunk = dict(self.chunks[idx])
            current = candidates.get(
                idx,
                {
                    **chunk,
                    "vector_score": 0.0,
                    "vector_score_norm": 0.0,
                },
            )
            current["bm25_score"] = score
            current["bm25_score_norm"] = norm_score
            candidates[idx] = current

        ranked_chunks: list[dict] = []
        for idx, item in candidates.items():
            domain_bonus = compute_domain_bonus(query, query_type, item)
            final_score = (
                0.50 * item["vector_score_norm"]
                + 0.30 * item["bm25_score_norm"]
                + 0.20 * domain_bonus
            )
            item["domain_bonus"] = domain_bonus
            item["final_score"] = float(final_score)
            ranked_chunks.append(item)

        # Light second-stage rerank: boost chunks whose vocabulary overlaps the query (no extra model).
        rerank_alpha = 0.1
        for item in ranked_chunks:
            ov = _token_overlap_ratio(query, item.get("text", ""))
            item["lexical_overlap"] = float(ov)
            item["final_score"] = float(min(1.0, item["final_score"] + rerank_alpha * ov))

        ranked_chunks.sort(key=lambda c: c["final_score"], reverse=True)
        deduped = self._dedupe(ranked_chunks)[:top_k]

        return {
            "query_type": query_type,
            "retrieved_chunks": deduped,
        }

    def _dedupe(self, chunks: list[dict]) -> list[dict]:
        seen_texts: set[str] = set()
        unique_chunks: list[dict] = []
        for chunk in chunks:
            signature = chunk["text"][:220]
            if signature not in seen_texts:
                seen_texts.add(signature)
                unique_chunks.append(chunk)
        return unique_chunks
