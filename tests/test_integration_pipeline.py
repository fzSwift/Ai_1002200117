import json

from src.generation.offline_answer import generate_offline_answer
from src.generation.prompt_builder import UNKNOWN_FROM_DOCUMENTS
from src.pipeline.rag_pipeline import OUT_OF_SCOPE_RESPONSE, RAGPipeline, _should_block_out_of_scope


class _FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def generate(self, prompt: str, query: str = "", chunks: list[dict] | None = None) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "query": query,
                "chunks": chunks or [],
            }
        )
        return self.answer


def test_pipeline_answer_writes_jsonl_and_returns_payload(monkeypatch, tmp_path) -> None:
    prep = {
        "query": "Who won?",
        "effective_query": "who won ghana election",
        "query_type": "election",
        "retrieved_chunks": [
            {
                "chunk_id": "csv_1",
                "source": "election_csv",
                "text": "Candidate A won.",
                "final_score": 0.93,
            }
        ],
        "final_prompt": "Prompt body",
    }
    llm = _FakeLLM("Candidate A won the election.")
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.log_file = tmp_path / "logs.jsonl"
    pipeline.llm = llm

    monkeypatch.setattr(RAGPipeline, "prepare_retrieval", lambda self, query, top_k=4, prompt_version="v3": prep)

    result = pipeline.answer("Who won?", top_k=4, prompt_version="v3")

    assert result["response"] == "Candidate A won the election."
    assert llm.calls and llm.calls[0]["query"] == "Who won?"
    lines = pipeline.log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["query"] == "Who won?"
    assert payload["response"] == "Candidate A won the election."
    assert "timestamp_utc" in payload


def test_pipeline_answer_supports_offline_abstention_response(monkeypatch, tmp_path) -> None:
    prep = {
        "query": "Tell me about 1992 values",
        "effective_query": "tell me about 1992 values",
        "query_type": "budget",
        "retrieved_chunks": [],
        "final_prompt": "Prompt body",
    }
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.log_file = tmp_path / "logs.jsonl"
    pipeline.llm = _FakeLLM(UNKNOWN_FROM_DOCUMENTS)

    monkeypatch.setattr(RAGPipeline, "prepare_retrieval", lambda self, query, top_k=4, prompt_version="v3": prep)

    result = pipeline.answer("Tell me about 1992 values")

    assert result["response"] == UNKNOWN_FROM_DOCUMENTS
    log_entry = json.loads(pipeline.log_file.read_text(encoding="utf-8").strip())
    assert log_entry["response"] == UNKNOWN_FROM_DOCUMENTS


def test_pipeline_pure_llm_answer_delegates_to_client() -> None:
    class _PureLLM:
        def pure_llm_answer(self, query: str) -> str:
            return f"pure: {query}"

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.llm = _PureLLM()

    assert pipeline.pure_llm_answer("test q") == "pure: test q"


def test_out_of_scope_gate_blocks_weak_irrelevant_queries() -> None:
    chunks = [
        {
            "chunk_id": "budget_0001",
            "source": "budget_pdf",
            "text": "Fiscal deficit guidance appears in budget annex tables.",
            "final_score": 0.12,
        }
    ]
    assert _should_block_out_of_scope("what is the weather in paris", chunks)


def test_out_of_scope_gate_blocks_world_cup_question() -> None:
    chunks = [
        {
            "chunk_id": "budget_para_p120_101",
            "source": "budget_pdf",
            "text": "The Department will continue workplace inspections and support job placement.",
            "final_score": 0.95,
        }
    ]
    assert _should_block_out_of_scope("who won the world cup", chunks)


def test_out_of_scope_gate_allows_in_domain_query_even_if_short() -> None:
    chunks = [
        {
            "chunk_id": "election_0001",
            "source": "election_csv",
            "text": "Election record. Year: 2020. New Region: Northern Region. Candidate: Nana Akufo Addo.",
            "final_score": 0.20,
        }
    ]
    assert not _should_block_out_of_scope("who won election?", chunks)


def test_pipeline_answer_uses_out_of_scope_override(monkeypatch, tmp_path) -> None:
    prep = {
        "query": "what is football score today",
        "effective_query": "what is football score today",
        "query_type": "mixed",
        "retrieved_chunks": [],
        "final_prompt": "OUT_OF_SCOPE_QUERY_PATH",
        "response_override": OUT_OF_SCOPE_RESPONSE,
        "confidence": "low",
        "citations": [],
        "router_path": "out_of_scope_block_path",
        "retrieval_constraints": {},
    }
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.log_file = tmp_path / "logs.jsonl"
    pipeline.llm = _FakeLLM("should-not-be-used")

    monkeypatch.setattr(RAGPipeline, "prepare_retrieval", lambda self, query, top_k=4, prompt_version="v3": prep)

    result = pipeline.answer("what is football score today")
    assert result["response"] == OUT_OF_SCOPE_RESPONSE
    assert pipeline.llm.calls == []


def test_offline_answer_handles_party_region_vote_query() -> None:
    chunks = [
        {
            "source": "election_csv",
            "text": (
                "Election record. Year: 2020. New Region: Northern Region. Code: NRT. "
                "Candidate: Nana Akufo Addo. Party: NPP. Votes: 487,260. Vote Percentage: 54.8%."
            ),
        },
        {
            "source": "election_csv",
            "text": (
                "Election record. Year: 2020. New Region: Northern Region. Code: NRT. "
                "Candidate: John Mahama. Party: NDC. Votes: 473,000. Vote Percentage: 45.2%."
            ),
        },
    ]
    answer = generate_offline_answer("What is the NPP vote in the nothen region?", chunks)
    assert "487,260" in answer
    assert "NPP" in answer
    assert "Northern Region" in answer
