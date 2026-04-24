from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.scoring import (
    classify_query,
    compute_domain_bonus,
    extract_metadata_constraints,
    metadata_match_score,
    normalize_scores,
)
from src.retrieval.vector_store import VectorStore


def _token_overlap_ratio(query: str, text: str) -> float:
    q = set(query.lower().split())
    if not q:
        return 0.0
    t = set(text.lower().split())
    return len(q & t) / len(q)


class HybridRetriever:
    def __init__(self, chunks: list[dict], *, cache_dir: str | Path | None = None) -> None:
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_hash = hashlib.sha1("||".join(self.texts[:2000]).encode("utf-8")).hexdigest()[:12]
        self.embedder = Embedder()
        self.embeddings, self.vector_store = self._load_or_build_index()
        self.bm25 = BM25Retriever(self.texts)
        self._cross_encoder: CrossEncoder | None = None

    def _load_or_build_index(self) -> tuple[np.ndarray, VectorStore]:
        if self.cache_dir is None:
            embeddings = self.embedder.encode(self.texts)
            store = VectorStore(embeddings.shape[1])
            store.add(embeddings)
            return embeddings, store

        emb_path = self.cache_dir / f"embeddings_{self.dataset_hash}.npy"
        idx_path = self.cache_dir / f"faiss_{self.dataset_hash}.index"
        if emb_path.exists() and idx_path.exists():
            embeddings = np.load(emb_path)
            store = VectorStore.load(idx_path)
            return embeddings, store

        embeddings = self.embedder.encode(self.texts)
        store = VectorStore(embeddings.shape[1])
        store.add(embeddings)
        np.save(emb_path, embeddings)
        store.save(idx_path)
        return embeddings, store

    def _get_cross_encoder(self) -> CrossEncoder | None:
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._cross_encoder = None
        return self._cross_encoder

    def _cross_rerank(self, query: str, chunks: list[dict], keep_top_k: int) -> list[dict]:
        model = self._get_cross_encoder()
        if model is None or not chunks:
            return chunks[:keep_top_k]
        pairs = [(query, c.get("text", "")) for c in chunks]
        try:
            scores = model.predict(pairs)
        except Exception:
            return chunks[:keep_top_k]
        for c, s in zip(chunks, scores):
            c["cross_score"] = float(s)
        chunks.sort(key=lambda c: c.get("cross_score", -1e9), reverse=True)
        return chunks[:keep_top_k]

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
        constraints = extract_metadata_constraints(qc)
        query_embedding = self.embedder.encode([query])[0]

        effective_initial_k = max(initial_k, 10)
        vector_scores, vector_indices = self.vector_store.search(query_embedding, top_k=effective_initial_k)
        bm25_scores, bm25_indices = self.bm25.search(query, top_k=effective_initial_k)

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
            meta_score = metadata_match_score(item, constraints)
            final_score = (
                0.50 * item["vector_score_norm"]
                + 0.30 * item["bm25_score_norm"]
                + 0.20 * domain_bonus
            )
            item["domain_bonus"] = domain_bonus
            item["metadata_match"] = meta_score
            final_score += 0.15 * meta_score
            item["final_score"] = float(final_score)
            ranked_chunks.append(item)

        # Light second-stage rerank: boost chunks whose vocabulary overlaps the query (no extra model).
        rerank_alpha = 0.1
        for item in ranked_chunks:
            ov = _token_overlap_ratio(query, item.get("text", ""))
            item["lexical_overlap"] = float(ov)
            item["final_score"] = float(min(1.0, item["final_score"] + rerank_alpha * ov))

        ranked_chunks.sort(key=lambda c: c["final_score"], reverse=True)
        deduped = self._dedupe(ranked_chunks)
        rerank_pool = deduped[:10]
        reranked = self._cross_rerank(query, rerank_pool, keep_top_k=min(3, top_k))
        tail_needed = max(0, top_k - len(reranked))
        if tail_needed:
            selected_ids = {c.get("chunk_id") for c in reranked}
            tail = [c for c in deduped if c.get("chunk_id") not in selected_ids][:tail_needed]
            deduped = reranked + tail
        else:
            deduped = reranked

        return {
            "query_type": query_type,
            "retrieved_chunks": deduped[:top_k],
            "constraints": constraints,
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
