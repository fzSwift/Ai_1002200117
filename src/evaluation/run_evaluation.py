from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.adversarial_tests import ALL_EVAL_QUERIES
from src.generation.prompt_builder import UNKNOWN_FROM_DOCUMENTS
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.helpers import OUTPUT_DIR, ensure_output_dir


def abstention_detected(text: str) -> bool:
    """True when the model explicitly declines (grounded abstention), not a numeric hallucination score."""
    t = (text or "").strip()
    if UNKNOWN_FROM_DOCUMENTS in t:
        return True
    return "i do not know" in t.lower()


def run() -> list[dict]:
    ensure_output_dir()
    pipeline = RAGPipeline()
    results: list[dict] = []

    for query in ALL_EVAL_QUERIES:
        rag_result = pipeline.answer(query=query, top_k=4, prompt_version="v3")
        pure_llm = pipeline.pure_llm_answer(query)

        results.append(
            {
                "query": query,
                "effective_query": rag_result.get("effective_query", query),
                "rag_answer": rag_result["response"],
                "pure_llm_answer": pure_llm,
                "query_type": rag_result["query_type"],
                "retrieved_chunk_ids": [c["chunk_id"] for c in rag_result["retrieved_chunks"]],
                "abstention_detected": abstention_detected(rag_result["response"]),
                "retrieval_quality_manual": "To be filled manually",
                "accuracy_manual": "To be filled manually",
                "hallucination_manual": "To be filled manually",
                "consistency_manual": "To be filled manually",
            }
        )

    out_path = OUTPUT_DIR / "evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
