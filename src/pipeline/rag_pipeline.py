from __future__ import annotations

import uuid
from time import perf_counter
from pathlib import Path

from src.data.structured_store import StructuredElectionStore
from src.generation.answer_composer import compose_structured_answer
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_prompt
from src.ingestion.load_csv import load_election_csv
from src.ingestion.load_pdf import load_pdf_text
from src.preprocessing.chunking import (
    chunk_election_records,
    chunk_pdf_fixed,
    chunk_pdf_paragraph_aware,
)
from src.preprocessing.clean_csv import clean_election_df
from src.preprocessing.clean_pdf import clean_pdf_pages
from src.routing.query_router import is_structured_numeric_query, parse_structured_constraints
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.scoring import has_domain_signal
from src.utils.helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir
from src.utils.logger import append_json_log


OUT_OF_SCOPE_RESPONSE = "I don't understand that from the provided documents."


def _dynamic_top_k_for_query(query: str, user_top_k: int) -> int:
    q = query.lower()
    if any(k in q for k in ("compare", "difference", "higher", "lower", "versus", "vs")):
        return max(user_top_k, 6)
    if any(k in q for k in ("how many", "how much", "total", "number", "votes", "percent", "percentage")):
        return max(user_top_k, 5)
    return max(user_top_k, 4)


def _token_overlap_ratio(query: str, text: str) -> float:
    q_terms = {t for t in query.lower().split() if len(t) > 2}
    if not q_terms:
        return 0.0
    text_terms = set(text.lower().split())
    return len(q_terms & text_terms) / len(q_terms)


def _should_block_out_of_scope(query: str, chunks: list[dict]) -> bool:
    if not chunks:
        return True
    if has_domain_signal(query):
        return False
    top_score = float(chunks[0].get("final_score", 0.0))
    best_overlap = max((_token_overlap_ratio(query, str(c.get("text", ""))) for c in chunks[:3]), default=0.0)
    return top_score < 0.40 and best_overlap < 0.15


class RAGPipeline:
    def __init__(self) -> None:
        ensure_output_dir()
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.log_file = self.output_dir / "logs.jsonl"
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.data_dir / "Ghana_Election_Result.csv"
        pdf_path = self.data_dir / "2025_budget.pdf"

        election_df = clean_election_df(load_election_csv(csv_path))
        pdf_pages = clean_pdf_pages(load_pdf_text(pdf_path))

        self.election_chunks = chunk_election_records(election_df)
        # Fixed-window PDF chunks are kept for experiments / ablations; the live index uses paragraph-aware only.
        self.pdf_fixed_chunks = chunk_pdf_fixed(pdf_pages)
        self.pdf_paragraph_chunks = chunk_pdf_paragraph_aware(pdf_pages)

        self.all_chunks = self.election_chunks + self.pdf_paragraph_chunks
        self.retriever = HybridRetriever(self.all_chunks, cache_dir=self.cache_dir)
        self.llm = LLMClient()
        self.structured_store = StructuredElectionStore(self.output_dir / "election_store.sqlite3")
        self.structured_store.build_from_df(election_df)

    def _structured_route(self, query: str) -> dict | None:
        if not is_structured_numeric_query(query):
            return None
        constraints = parse_structured_constraints(query)
        rows = self.structured_store.query_votes(
            party=constraints.get("party"),
            region=constraints.get("region"),
            year=constraints.get("year"),
            limit=3,
        )
        if not rows:
            return None

        top = rows[0]
        chunk_id = f"election_{int(top['row_index']):04d}"
        answer = (
            f"Direct answer: {top.get('Party','')} in {top.get('new_region') or top.get('old_region')} has "
            f"{int(top.get('Votes', 0)):,} votes for {top.get('Candidate')} in {top.get('Year')} "
            f"({top.get('votes_pct')}). [{chunk_id}]"
        )
        evidence = (
            f"Evidence: Candidate={top.get('Candidate')}, Party={top.get('Party')}, Region={top.get('new_region')}, "
            f"Votes={int(top.get('Votes', 0)):,}. [{chunk_id}]"
        )
        response = f"{answer}\n{evidence}\nConfidence: high."

        retrieved_chunks: list[dict] = []
        for row in rows:
            cid = f"election_{int(row['row_index']):04d}"
            match = next((c for c in self.election_chunks if c.get("chunk_id") == cid), None)
            if match:
                chunk = dict(match)
                chunk["final_score"] = 1.0
                chunk["structured_match"] = True
                retrieved_chunks.append(chunk)

        return {
            "query_type": "election",
            "retrieved_chunks": retrieved_chunks,
            "final_prompt": "STRUCTURED_QUERY_PATH",
            "effective_query": query,
            "response_override": response,
            "confidence": "high",
            "citations": [c["chunk_id"] for c in retrieved_chunks],
            "router_path": "structured_query_path",
        }

    def prepare_retrieval(self, query: str, top_k: int = 4, prompt_version: str = "v3") -> dict:
        """Run query rewrite + hybrid retrieval + prompt build. Used by `answer` and the Streamlit stream path."""
        q = query.strip()
        route = self._structured_route(q)
        if route is not None:
            return {
                "query": q,
                "effective_query": route["effective_query"],
                "query_type": route["query_type"],
                "retrieved_chunks": route["retrieved_chunks"][:top_k],
                "final_prompt": route["final_prompt"],
                "response_override": route["response_override"],
                "confidence": route["confidence"],
                "citations": route["citations"],
                "router_path": route["router_path"],
                "retrieval_constraints": {},
            }

        effective_query = rewrite_query(q)
        effective_top_k = _dynamic_top_k_for_query(q, top_k)
        retrieval_result = self.retriever.retrieve(
            query=effective_query,
            top_k=effective_top_k,
            classification_query=q,
        )
        selected_chunks = retrieval_result["retrieved_chunks"]
        if _should_block_out_of_scope(q, selected_chunks):
            return {
                "query": q,
                "effective_query": effective_query,
                "query_type": retrieval_result["query_type"],
                "retrieved_chunks": selected_chunks,
                "final_prompt": "OUT_OF_SCOPE_QUERY_PATH",
                "response_override": OUT_OF_SCOPE_RESPONSE,
                "confidence": "low",
                "citations": [],
                "router_path": "out_of_scope_block_path",
                "retrieval_constraints": retrieval_result.get("constraints", {}),
            }
        final_prompt = build_prompt(query=q, chunks=selected_chunks, version=prompt_version)
        return {
            "query": q,
            "effective_query": effective_query,
            "query_type": retrieval_result["query_type"],
            "retrieved_chunks": selected_chunks,
            "final_prompt": final_prompt,
            "retrieval_constraints": retrieval_result.get("constraints", {}),
            "router_path": "rag_narrative_path",
        }

    def finalize_answer(self, prep: dict, response: str, *, timings: dict[str, float] | None = None, request_id: str | None = None) -> dict:
        composed = compose_structured_answer(prep["query"], prep["retrieved_chunks"], mode=getattr(self.llm, "response_mode", "detailed"))
        confidence = prep.get("confidence", composed.confidence)
        citations = prep.get("citations", composed.citations)
        payload = {
            "request_id": request_id or str(uuid.uuid4()),
            "query": prep["query"],
            "effective_query": prep["effective_query"],
            "query_type": prep["query_type"],
            "chunking_strategy_used": "record-based + paragraph-aware",
            "retrieved_chunks": prep["retrieved_chunks"],
            "final_prompt": prep["final_prompt"],
            "response": response,
            "response_intent": composed.intent,
            "confidence": confidence,
            "citations": citations,
            "router_path": prep.get("router_path", "rag_narrative_path"),
            "retrieval_constraints": prep.get("retrieval_constraints", {}),
            "timings_ms": timings or {},
        }
        log_path = append_json_log(self.log_file, payload)
        return {**payload, "log_path": log_path}

    def answer(self, query: str, top_k: int = 4, prompt_version: str = "v3") -> dict:
        t0 = perf_counter()
        request_id = str(uuid.uuid4())
        prep = self.prepare_retrieval(query, top_k=top_k, prompt_version=prompt_version)
        t1 = perf_counter()
        if prep.get("response_override"):
            return self.finalize_answer(
                prep,
                prep["response_override"],
                timings={
                    "rewrite_retrieve_ms": round((t1 - t0) * 1000, 2),
                    "generate_ms": 0.0,
                    "total_ms": round((perf_counter() - t0) * 1000, 2),
                },
                request_id=request_id,
            )

        g0 = perf_counter()
        response = self.llm.generate(
            prep["final_prompt"],
            query=prep["query"],
            chunks=prep["retrieved_chunks"],
        )
        g1 = perf_counter()
        return self.finalize_answer(
            prep,
            response,
            timings={
                "rewrite_retrieve_ms": round((t1 - t0) * 1000, 2),
                "generate_ms": round((g1 - g0) * 1000, 2),
                "total_ms": round((g1 - t0) * 1000, 2),
            },
            request_id=request_id,
        )

    def pure_llm_answer(self, query: str) -> str:
        return self.llm.pure_llm_answer(query)
