# FinanceBench RAG System & Retrieval Ablation Benchmark (UNDER DEVELOPMENT)

This repository contains a production-grade **Retrieval-Augmented Generation (RAG)** pipeline and automated evaluation benchmark on the **FinanceBench** benchmark suite (SEC 10-K filings for AMD, AMCOR, American Express, Boeing, and PepsiCo) built with Python 3.12 and `uv`.

---

## 🏛️ System Architecture Overview

```
                                 User Query / Question
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [Baseline Pipeline]                          [Production Pipeline]
                    │                                             │
      Dense Vector Search (Top-5)                    Contextual Metadata Pre-filter
     (Naive Fixed-Window Chunks)                                  │
                    │                               ┌─────────────┴─────────────┐
                    │                               ▼                           ▼
                    │                     Dense Vector Search           BM25 Lexical Search
                    │                      (Top-20 via Qdrant)         (Top-20 via Rank-BM25)
                    │                               │                           │
                    │                               └─────────────┬─────────────┘
                    │                                             ▼
                    │                               Reciprocal Rank Fusion (RRF, k=60)
                    │                                             │
                    │                               FlashRank Cross-Encoder Reranking
                    │                                  (ms-marco-TinyBERT-L-2-v2)
                    │                                             │
                    │                               Top-5 Calibrated Blended Passages
                    │                                             │
                    │                              Score Check: >= 0.30 Threshold?
                    │                                     /                \
                    │                                   YES                NO
                    │                                    │                  │
                    │                                    ▼                  ▼
                    │                              LLM Generation   Abstention Guardrail
                    │                                    │           (is_grounded=False)
                    │                       (Gemini -> Groq Fallback)       │
                    ▼                                    │                  │
             Baseline Answer                             └──────────┬───────┘
                                                                    ▼
                                                            Production Answer
```

---

## 📦 Key Components Implemented

### 1. Configuration & Resilience ([`src/config.py`](src/config.py))
- Managed via `pydantic-settings` with `.env` loading.
- Primary LLM: `gemini-3-flash-preview` (Gemini API via `google-genai`).
- Primary Embeddings: `gemini-embedding-2` (3,072 dimensions) with fallback to `gemini-embedding-001`.
- Automatic LLM Fallback: `openai/gpt-oss-120b` (Groq API) triggered on HTTP 429 (rate limits) or service degradation.
- Strict retrieval parameters: `TOP_K_RETRIEVAL = 20`, `TOP_K_RERANK = 5`, `RERANK_SCORE_THRESHOLD = 0.30`, and `RRF_K = 60`.

### 2. Ingestion & Structured Parsing ([`src/ingestion/parser_and_chunker.py`](src/ingestion/parser_and_chunker.py))
- Extracted all 5 SEC filings page-by-page preserving Markdown tables using `pymupdf4llm(..., page_chunks=True, use_ocr=False)`.
- Pre-parsed and cached 1,230 pages across 5 SEC 10-K filings into JSON (`data/cache/parsed_pages/`).
- Dual chunking modes for ablation testing:
  - `naive_chunk()`: Fixed character splitting (1,000 characters, 100 character overlap) without document headers.
  - `contextual_markdown_chunk()`: Section splitting respecting table boundaries, with prepended context headers: `"[Document: {doc_name} | Page: {page_num}]\n{chunk_text}"`.

### 3. Local Vector & Lexical Retrieval ([`src/retrieval/hybrid_retriever.py`](src/retrieval/hybrid_retriever.py))
- **`EmbeddingManager`**: SQLite-backed embedding cache (`data/cache/embedding_cache.sqlite`) storing 3,072-dim embeddings. Ensures 0 re-computations or quota consumption for previously seen chunks and queries.
- **`QdrantVectorStore`**: Local vector storage at `./qdrant_storage` with collections `baseline_chunks` and `production_chunks`. Closes client sessions per operation to eliminate local file lock contention.
- **`BM25IndexManager`**: Serialized `rank_bm25.BM25Okapi` inverted index (`data/bm25_indices/production_bm25.pkl`) with company/document pre-filtering.
- **`HybridRetriever`**:
  - Baseline: Pure cosine vector search (top-5).
  - Production: Hybrid retrieval fusing Top-20 Dense Vector hits and Top-20 BM25 hits via Reciprocal Rank Fusion ($k=60$), followed by FlashRank `ms-marco-TinyBERT-L-2-v2` cross-encoder reranking. Scores are calibrated and blended 50/50 with dense similarity.

### 4. Grounded Generation & Guardrails ([`src/generation/generator.py`](src/generation/generator.py))
- Strict Pydantic structured output contract:
  ```python
  class RAGResponse(BaseModel):
      answer: str
      cited_pages: list[int]
      cited_doc_names: list[str]
      is_grounded: bool
      provider_used: str
  ```
- **Abstention Guardrail**: If the top reranker score is below `0.30`, skips LLM execution and immediately returns `is_grounded=False` with `provider_used="abstention_guardrail"`.
- Seamless failover between Google GenAI and Groq API.

### 5. Automated Retrieval Ablation Benchmark ([`src/evaluation/evaluate.py`](src/evaluation/evaluate.py))
- Automated test runner over all 30 FinanceBench ground-truth Q&A pairs in [`data/eval_dataset.json`](data/eval_dataset.json).
- Deterministic evaluation of **Document Hit Rate @ 5**, **Exact Page Hit Rate @ 5**, and **Mean Reciprocal Rank (MRR@5)** without consuming LLM token quota.
- Automatic reporting to [`data/ablation_results.md`](data/ablation_results.md) and [`data/ablation_results.json`](data/ablation_results.json).

### 6. FastAPI Backend & Streamlit Web UI
- **FastAPI ([`api/main.py`](api/main.py))**:
  - `GET /health`: Health check, storage stats, and active config.
  - `POST /ingest`: Triggers document parsing and vector upsert.
  - `POST /query`: Single pipeline execution (`baseline` or `production`).
  - `POST /query/compare`: Side-by-side execution on the same query.
- **Streamlit ([`app/streamlit_app.py`](app/streamlit_app.py))**:
  - Dropdown selector for all 30 FinanceBench questions (with company filtering).
  - Live side-by-side comparator: Baseline vs. Production RAG (Latency, Groundedness badges, Citations, Provider).
  - Expandable chunk inspector displaying Top-5 passages with reranker and RRF scores.
  - FinanceBench ground truth comparison card with retrieval hit verification.
  - Interactive benchmark dashboard tab showing ablation results.

---

## 📊 Retrieval Ablation Benchmark Results

*Evaluated on the 30 ground-truth questions in FinanceBench:*

| Metric | Baseline (Dense Only) | Production (Hybrid + RRF + Rerank) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Exact Page Hit Rate @ 5** | **60.0%** | **60.0%** | Matches baseline |
| **Mean Reciprocal Rank (MRR@5)** | **0.4944** | **0.5000** | **+1.1% gain** |
| **Document Hit Rate @ 5** | **90.0%** | **100.0%** | **+11.1% gain (100% accuracy)** |
| **Avg Retrieval Latency** | **146.8 ms** | **293.1 ms** | Sub-300ms latency |

### Per-Company Evidence Retrieval Highlights:
- **`AMCOR_2023_10K`**: Baseline Page Hit was **0.0%** (failed on table queries). Production Hybrid achieved **50.0% Page Hit** and **0.312 MRR**.
- **`AMERICANEXPRESS_2022_10K`**: Production MRR reached **0.714** (vs. 0.500 for baseline).
- **`AMD_2022_10K`**: **100% Document Hit** and **85.7% Page Hit @ 5**.

---

## 🚀 Running the System

### 1. CLI Commands (`main.py`)
```bash
# Query the production RAG pipeline (Gemini primary)
uv run python main.py query "What drove revenue change as of the FY22 for AMD?" --doc-name AMD_2022_10K

# Query using Groq explicitly
uv run python main.py query "What drove revenue change as of the FY22 for AMD?" --doc-name AMD_2022_10K --provider groq

# Run baseline dense search
uv run python main.py query "What drove revenue change as of the FY22 for AMD?" --mode baseline

# Run the 30-query retrieval ablation benchmark
uv run python main.py evaluate --use-doc-filter
```

### 2. FastAPI Backend
```bash
# Start FastAPI server on port 8000
uv run python main.py api --port 8000

# Or directly with uvicorn
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Streamlit Visual Dashboard
```bash
# Launch interactive Streamlit UI on port 8501
uv run python main.py app
# or
uv run streamlit run app/streamlit_app.py --server.port 8501
```
