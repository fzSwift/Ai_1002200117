# AcaIntel AI — Academic City RAG (CS4241)

**Student Name:** Chidiebele Benjamin Amechi  
**Index Number:** 10022200117

## Overview

AcaIntel AI is a custom **Retrieval-Augmented Generation (RAG)** system built for the CS4241 project exam.  
It answers user questions using two local sources:

1. `data/Ghana_Election_Result.csv` (structured election data)
2. `data/2025_budget.pdf` (long-form budget document)

This project is implemented from scratch in `src/` and does **not** rely on end-to-end RAG frameworks like LangChain or LlamaIndex.

---

## What Makes This Project Strong

- **Hybrid retrieval**: FAISS (dense) + BM25 (lexical) with score fusion
- **Query rewrite**: expands short or ambiguous phrasing before retrieval
- **Domain-aware ranking**: election-like queries favor CSV chunks; budget-like queries favor PDF chunks
- **Grounded offline QA**: no API key required; still returns synthesized answers
- **Transparent UI**: shows retrieved chunks, scores, effective query, and prompt
- **Mode flexibility**: supports Offline, OpenAI, Ollama Local, and Ollama Cloud generation
- **Multi-chunk answer synthesis**: weighted fact voting across retrieved chunks for numeric/comparison answers
- **Reliability guardrails**: contradiction detection with confidence penalty when evidence conflicts

---

## System Architecture

### Core modules

- `src/ingestion/`: load CSV/PDF source files
- `src/preprocessing/`: cleaning and chunking
- `src/retrieval/`: embeddings, vector store, BM25, hybrid scoring and reranking
- `src/generation/`: prompt builder, model clients, offline answer synthesis
- `src/pipeline/`: end-to-end orchestration (`RAGPipeline`)
- `src/evaluation/`: evaluation query sets and runner
- `src/utils/`: logging and path helpers

### End-to-end flow

```text
User query
  -> query rewrite + alias normalization
  -> query classification (election / budget / mixed)
  -> intent router (structured numeric path vs narrative RAG path)
  -> FAISS retrieval + BM25 retrieval
  -> metadata-aware filtering boost (source/year/region/party)
  -> weighted fusion + domain bonus + lexical rerank
  -> cross-encoder rerank (top pool -> requested top-k)
  -> dynamic top-k by intent (comparison/numeric pull deeper evidence)
  -> dedupe + top-k chunk selection
  -> answer composer (direct answer + evidence + confidence + contradiction penalty)
  -> response generation (offline / Ollama / OpenAI)
  -> JSONL logging with request_id + stage timings
```

Fusion rule:

```text
final_score = 0.50 * vector_norm + 0.30 * bm25_norm + 0.20 * domain_bonus
```

---

## Chunking Strategy

1. **Record-based chunking for CSV**
   - Each row becomes a compact, structured chunk
   - Ideal for factual, numeric, and comparison questions

2. **Paragraph-aware chunking for PDF**
   - Splits by semantic paragraph boundaries
   - Better context continuity for policy/fiscal explanations

Note: fixed-window PDF chunks are retained for experimentation, but live retrieval uses paragraph-aware PDF chunks plus election chunks.

---

## Generation Modes

The app supports three answer modes:

1. **Offline mode (free, default fallback)**
   - Works when `OPENAI_API_KEY` is empty or `OFFLINE_MODE=1`
   - Produces grounded synthesized answers from retrieved chunks
   - Includes numeric refinement logic (vote/percentage extraction and comparative phrasing)

2. **Ollama local mode (free local model)**
   - Uses local model served at `OLLAMA_BASE_URL` (or `OLLAMA_LOCAL_BASE_URL`)
   - Supports full generation and token streaming

3. **Ollama cloud mode**
   - Uses cloud endpoint at `https://ollama.com` (or `OLLAMA_CLOUD_BASE_URL`)
   - Requires `OLLAMA_API_KEY` or `OLLAMA_CLOUD_API_KEY`
   - Can be toggled at runtime from the Streamlit sidebar (`Ollama Local` / `Ollama Cloud`)

4. **OpenAI API mode**
   - Uses `OPENAI_API_KEY`
   - Supports standard chat generation and streaming

---

## Streamlit UI Features

Run:

```bash
streamlit run app.py
```

Main features:

- Chat-based question answering
- New chat + past conversation archive
- Export current thread (`.md` and `.json`)
- Evidence panel with retrieved chunks and scoring details
- Confidence indicator (low/medium/high)
- Sources used section with chunk previews
- A/B panel toggle (RAG answer vs pure LLM answer)
- Answer mode toggle (`concise`, `detailed`, `examiner`)
- Ollama target toggle (`Ollama Local` / `Ollama Cloud`) in sidebar
- Prompt visibility for explainability
- Light/dark UI and response rendering options

The app includes user-safe error handling for:
- pipeline initialization failures
- retrieval failures
- generation failures

---

## Setup

### 1) Create and activate virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

Copy env template:

- Windows PowerShell:
  - `Copy-Item .env.example .env`
- macOS/Linux:
  - `cp .env.example .env`

Edit `.env` according to your preferred mode.

---

## `.env` Configuration Examples

### Offline mode (free)

```env
OPENAI_API_KEY=
# OFFLINE_MODE=1
```

### Ollama local mode (free)

```env
USE_OLLAMA=1
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_API_KEY=
OPENAI_API_KEY=
```

### Ollama cloud mode

```env
USE_OLLAMA=1
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key
OPENAI_API_KEY=
```

### Ollama local + cloud dual setup (recommended)

```env
USE_OLLAMA=1

OLLAMA_LOCAL_BASE_URL=http://127.0.0.1:11434
OLLAMA_LOCAL_MODEL=llama3.1:8b
OLLAMA_LOCAL_API_KEY=

OLLAMA_CLOUD_BASE_URL=https://ollama.com
OLLAMA_CLOUD_MODEL=gpt-oss:120b-cloud
OLLAMA_CLOUD_API_KEY=your_ollama_api_key

# Optional default selection (auto-detected from URL)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
```

### OpenAI mode (paid)

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
```

---

## Ollama Quick Start

Install Ollama (Windows PowerShell):

```powershell
irm https://ollama.com/install.ps1 | iex
```

Pull model:

```bash
ollama pull llama3.1:8b
```

Verify:

```bash
ollama list
```

---

## Evaluation

Run:

```bash
python -m src.evaluation.run_evaluation
```

Output: `outputs/evaluation_results.json`
Benchmark metrics: `outputs/benchmark_metrics.json`

Key fields include:
- `rag_answer`
- `pure_llm_answer`
- `query_type`
- `retrieved_chunk_ids`
- `effective_query`
- `abstention_detected`
- `confidence`
- `citations`
- `timings_ms`

Use manual rubric columns in the output JSON for final project scoring notes.

Regression benchmark:

- Benchmark dataset: `evaluation/benchmarks/core_benchmark.json`
- Metrics tracked:
  - exact match
  - numeric accuracy
  - abstention correctness
  - citation precision
- CI blocks regressions if key thresholds drop.

---

## Logging and Outputs

- Runtime logs: `outputs/logs.jsonl` (append-only JSON lines)
- Evaluation results: `outputs/evaluation_results.json`

Each log entry includes query, effective query, retrieved chunks, prompt, response, confidence, citations, request id, stage timings, and timestamp.

---

## Build and Cache Indexes

To prebuild retrieval artifacts (FAISS index, embeddings cache, structured election store):

```bash
python build_index.py
```

This reduces cold-start overhead in deployed environments.

---

## Tests

Run all tests:

```bash
python -m pytest tests/
```

Project currently includes retrieval, chunking, and pipeline integration tests.
Additional tests validate structured answer composition for numeric/comparison synthesis.

---

## Deploy on Streamlit Community Cloud

### 1) Push code to GitHub

Your repo is already on GitHub:

- `https://github.com/fzSwift/Ai_1002200117`

### 2) Create the app on Streamlit

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Select:
   - **Repository**: `fzSwift/Ai_1002200117`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**.

### 3) Configure Secrets (recommended)

In Streamlit app settings, open **Secrets** and add only what you need.

Offline-only mode:

```toml
OFFLINE_MODE = "1"
OPENAI_API_KEY = ""
```

OpenAI mode:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4.1-mini"
```

Important:
- Do **not** use Ollama mode on Streamlit Cloud (`USE_OLLAMA=1`) because there is no local Ollama server in that environment.
- `runtime.txt` is pinned to Python 3.11 for package compatibility (FAISS/sentence-transformers).

### 4) Redeploy after updates

- Push changes to `main` and Streamlit redeploys automatically.
- You can also trigger a manual reboot/redeploy from the app settings.

### 5) Common deployment fixes

- If build fails, check **Manage app -> Logs** first.
- If memory is tight, reduce retrieval `top_k` in UI defaults and avoid unnecessary large model behavior.
- Keep secrets only in Streamlit Secrets, not in `.env` committed files.

---

## Limitations

- Answers are constrained by CSV/PDF coverage
- Query rewrite is rule-based and may miss unusual phrasing
- Numeric interpretation may depend on row granularity in source data
- Offline mode is grounded and improved, but still less fluent than larger cloud LLMs

---

## Submission Notes

This system is designed to demonstrate:
- grounded retrieval-first QA
- explainability (evidence + prompt transparency)
- robust offline operation
- clean separation of ingestion, retrieval, generation, and evaluation components
