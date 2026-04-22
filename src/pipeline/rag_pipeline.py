from __future__ import annotations

from pathlib import Path

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
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewrite import rewrite_query
from src.utils.helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir
from src.utils.logger import append_json_log


class RAGPipeline:
    def __init__(self) -> None:
        ensure_output_dir()
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.log_file = self.output_dir / "logs.jsonl"

        csv_path = self.data_dir / "Ghana_Election_Result.csv"
        pdf_path = self.data_dir / "2025_budget.pdf"

        election_df = clean_election_df(load_election_csv(csv_path))
        pdf_pages = clean_pdf_pages(load_pdf_text(pdf_path))

        self.election_chunks = chunk_election_records(election_df)
        # Fixed-window PDF chunks are kept for experiments / ablations; the live index uses paragraph-aware only.
        self.pdf_fixed_chunks = chunk_pdf_fixed(pdf_pages)
        self.pdf_paragraph_chunks = chunk_pdf_paragraph_aware(pdf_pages)

        self.all_chunks = self.election_chunks + self.pdf_paragraph_chunks
        self.retriever = HybridRetriever(self.all_chunks)
        self.llm = LLMClient()

    def prepare_retrieval(self, query: str, top_k: int = 4, prompt_version: str = "v3") -> dict:
        """Run query rewrite + hybrid retrieval + prompt build. Used by `answer` and the Streamlit stream path."""
        q = query.strip()
        effective_query = rewrite_query(q)
        retrieval_result = self.retriever.retrieve(
            query=effective_query,
            top_k=top_k,
            classification_query=q,
        )
        selected_chunks = retrieval_result["retrieved_chunks"]
        final_prompt = build_prompt(query=q, chunks=selected_chunks, version=prompt_version)
        return {
            "query": q,
            "effective_query": effective_query,
            "query_type": retrieval_result["query_type"],
            "retrieved_chunks": selected_chunks,
            "final_prompt": final_prompt,
        }

    def finalize_answer(self, prep: dict, response: str) -> dict:
        payload = {
            "query": prep["query"],
            "effective_query": prep["effective_query"],
            "query_type": prep["query_type"],
            "chunking_strategy_used": "record-based + paragraph-aware",
            "retrieved_chunks": prep["retrieved_chunks"],
            "final_prompt": prep["final_prompt"],
            "response": response,
        }
        log_path = append_json_log(self.log_file, payload)
        return {**payload, "log_path": log_path}

    def answer(self, query: str, top_k: int = 4, prompt_version: str = "v3") -> dict:
        prep = self.prepare_retrieval(query, top_k=top_k, prompt_version=prompt_version)
        response = self.llm.generate(
            prep["final_prompt"],
            query=prep["query"],
            chunks=prep["retrieved_chunks"],
        )
        return self.finalize_answer(prep, response)

    def pure_llm_answer(self, query: str) -> str:
        return self.llm.pure_llm_answer(query)
