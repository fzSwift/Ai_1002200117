from __future__ import annotations

import re
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


def _extract_first_number(text: str) -> float | None:
    m = re.search(r"\b\d[\d,]*(?:\.\d+)?\b", text or "")
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def run_benchmark(
    pipeline: RAGPipeline,
    benchmark_path: str | Path = Path("evaluation/benchmarks/core_benchmark.json"),
) -> dict:
    p = Path(benchmark_path)
    if not p.exists():
        return {"cases": 0, "exact_match": 0.0, "numeric_accuracy": 0.0, "abstention_correctness": 0.0, "citation_precision": 0.0}

    cases = json.loads(p.read_text(encoding="utf-8"))
    exact_hits = 0
    numeric_hits = 0
    abstain_hits = 0
    citation_hits = 0

    for case in cases:
        result = pipeline.answer(case["query"], top_k=4, prompt_version="v3")
        answer = result["response"]
        cids = [c["chunk_id"] for c in result.get("retrieved_chunks", [])]

        if case.get("expected_phrase") and case["expected_phrase"].lower() in answer.lower():
            exact_hits += 1

        if case.get("expected_numeric_min") is not None:
            n = _extract_first_number(answer)
            if n is not None and n >= float(case["expected_numeric_min"]):
                numeric_hits += 1

        expected_abstain = bool(case.get("expect_abstain", False))
        if abstention_detected(answer) == expected_abstain:
            abstain_hits += 1

        prefix = case.get("expected_chunk_id_prefix")
        if prefix:
            if any(str(cid).startswith(prefix) for cid in cids):
                citation_hits += 1
        else:
            citation_hits += 1

    total = max(1, len(cases))
    return {
        "cases": len(cases),
        "exact_match": round(exact_hits / total, 3),
        "numeric_accuracy": round(numeric_hits / total, 3),
        "abstention_correctness": round(abstain_hits / total, 3),
        "citation_precision": round(citation_hits / total, 3),
    }


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
    benchmark = run_benchmark(pipeline)
    (OUTPUT_DIR / "benchmark_metrics.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
