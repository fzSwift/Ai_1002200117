from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%$¢]+", text.lower())


class BM25Retriever:
    def __init__(self, texts: Sequence[str]) -> None:
        self.corpus_tokens = [tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> tuple[list[float], list[int]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        indices = [idx for idx, _ in ranked]
        values = [float(score) for _, score in ranked]
        return values, indices
