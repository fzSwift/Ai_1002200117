from src.retrieval.scoring import classify_query, normalize_scores


def test_classify_query_budget() -> None:
    assert classify_query("What is the 2025 budget allocation for education?") == "budget"


def test_normalize_scores() -> None:
    values = normalize_scores([1.0, 2.0, 3.0])
    assert values[0] == 0.0
    assert values[-1] == 1.0
