from src.generation.answer_composer import compose_structured_answer


def test_comparison_query_uses_numeric_sentence_not_raw_record() -> None:
    chunks = [
        {
            "chunk_id": "election_0126",
            "final_score": 0.91,
            "text": (
                "Election record. Year: 2020. Old Region: Northern Region. New Region: North East Region. "
                "Code: NPP. Candidate: Nana Akufo Addo. Party: NPP. Votes: 122742. Vote Percentage: 51.33%."
            ),
        },
        {
            "chunk_id": "election_0120",
            "final_score": 0.88,
            "text": (
                "Election record. Year: 2020. Old Region: Northern Region. New Region: Savannah Region. "
                "Code: NPP. Candidate: Nana Akufo Addo. Party: NPP. Votes: 80605. Vote Percentage: 35.19%."
            ),
        },
    ]
    answer = compose_structured_answer(
        "Compare vote counts between North East and Savannah regions.",
        chunks,
        mode="detailed",
    )
    assert "Comparison result:" in answer.text
    assert "North East Region" in answer.text
    assert "Savannah Region" in answer.text
    assert "122,742" in answer.text
    assert "80,605" in answer.text


def test_numeric_weighted_voting_and_contradiction_penalty() -> None:
    chunks = [
        {
            "chunk_id": "election_1001",
            "final_score": 0.95,
            "text": (
                "Election record. Year: 2020. Old Region: Central Region. New Region: Central Region. "
                "Code: NDC. Candidate: Jerry John Rawlings. Party: NDC. Votes: 222092. Vote Percentage: 66.49%."
            ),
        },
        {
            "chunk_id": "election_1002",
            "final_score": 0.45,
            "text": (
                "Election record. Year: 2020. Old Region: Central Region. New Region: Central Region. "
                "Code: NDC. Candidate: Jerry John Rawlings. Party: NDC. Votes: 220000. Vote Percentage: 65.00%."
            ),
        },
    ]
    answer = compose_structured_answer("How many votes did NDC get in Central Region?", chunks, mode="detailed")
    assert "weighted votes" in answer.text
    assert answer.confidence in {"medium", "low"}
