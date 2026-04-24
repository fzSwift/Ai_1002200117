from __future__ import annotations

from src.evaluation.run_evaluation import abstention_detected, run_benchmark


class _FakePipeline:
    def answer(self, query: str, top_k: int = 4, prompt_version: str = "v3") -> dict:
        q = query.lower()
        if "npp vote in the northern region" in q:
            return {
                "response": "The retrieved records show NPP in Northern Region with 122,742 votes. [election_0001]",
                "retrieved_chunks": [{"chunk_id": "election_0001"}],
            }
        if "total nationwide turnout" in q:
            return {
                "response": "National turnout is not stored as a single row in the retrieved election records.",
                "retrieved_chunks": [{"chunk_id": "election_0005"}],
            }
        if "inflation" in q:
            return {
                "response": "The retrieved records report inflation declining in 2025. [budget_para_p22_012]",
                "retrieved_chunks": [{"chunk_id": "budget_para_p22_012"}],
            }
        return {
            "response": "I do not have enough information from the provided documents.",
            "retrieved_chunks": [],
        }


def test_abstention_detected_case_insensitive() -> None:
    assert abstention_detected("I do not know from these docs.")
    assert abstention_detected("I do not have enough information from the provided documents.")


def test_run_benchmark_metrics_shape() -> None:
    metrics = run_benchmark(_FakePipeline())
    assert metrics["cases"] > 0
    assert 0.0 <= metrics["abstention_correctness"] <= 1.0
    assert "citation_precision" in metrics
