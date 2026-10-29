import streamlit as st

st.title("System Architecture & Design Decisions", icon=":material/schema:")
st.caption(
    "Deep dive into the production retrieval-augmented generation (RAG) architecture engineered for SEC 10-K financial filings."
)

st.markdown("### 🏛️ End-to-End Pipeline Dataflow")

# Mermaid diagram
st.markdown(
    """
```mermaid
flowchart TD
    subgraph Ingestion ["Ingestion & Contextual Chunking"]
        PDF["20 SEC 10-K PDFs"] --> Parser["PyMuPDF Markdown Parser"]
        Parser --> Headers["Context Headers\n[Company | Year | Page | Section]"]
        Headers --> DenseIdx["FastEmbed Embeddings\n(bge-small-en-v1.5)"]
        Headers --> LexIdx["BM25 Indexer\n(Financial Stopwords)"]
        DenseIdx --> Qdrant[("Qdrant Vector DB\n(5,391 Chunks)")]
        LexIdx --> BM25File[("BM25 Inverted Pickle\n(5,391 Chunks)")]
    end

    subgraph Retrieval ["Hybrid Retrieval & Reranking"]
        Q["User Financial Query"] --> DenseSearch["Qdrant Vector Search\n(Top-25)"]
        Q --> BM25Search["BM25 Lexical Search\n(Top-25)"]
        DenseSearch & BM25Search --> RRF["Reciprocal Rank Fusion\n(k = 60)"]
        RRF --> Rerank["FlashRank Cross-Encoder\n(ms-marco-MiniLM-L-12-v2)"]
        Rerank --> Guard{"Max Score >= 0.30?"}
    end

    subgraph Generation ["Guarded LLM Generation"]
        Guard -- "No (< 0.30)" --> Abstain["Instant Abstention\n(No LLM Call / Zero Hallucination)"]
        Guard -- "Yes (>= 0.30)" --> Prompt["Grounding Prompt + Citations"]
        Prompt --> Gemini["Primary LLM\n(Gemini 2.5 Flash)"]
        Gemini -. "429 Rate Limit" .-> Groq["Fallback LLM\n(Groq Llama 3.3 70B)"]
        Gemini & Groq --> Output["Grounded Response + Page Citations"]
    end
```
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 1. Ingestion & Contextual Chunking")
        st.markdown(
            """
            - **PDF Extraction:** Parsed with `PyMuPDF` preserving markdown table formatting and page bounds.
            - **Contextual Chunking:** Every chunk is prepended with a metadata header:
              ```text
              [Document: AMD_2022_10K | Company: Advanced Micro Devices | Fiscal Year: 2022 | Page: 42 | Section: Item 7 MD&A]
              ```
            - **Why it matters:** In SEC 10-K filings, numerical tables frequently lack inline company or year mentions on subsequent pages. Context headers ensure dense embeddings and BM25 preserve document and temporal identity.
            """
        )

    with col2:
        with st.container(border=True):
            st.markdown("### 2. Dual-Index Hybrid Retrieval")
            st.markdown(
                """
                - **Dense Path:** `BAAI/bge-small-en-v1.5` (384-dimensional embeddings) indexed in local Qdrant collections.
                - **Lexical Path:** BM25Okapi inverted index with specialized stopword filtering (stripping 80+ generic legal/financial filler terms).
                - **Reciprocal Rank Fusion (RRF):**
                  $$RRF(d) = \\frac{1}{60 + r_{dense}(d)} + \\frac{1}{60 + r_{bm25}(d)}$$
                - **Why it matters:** Dense vectors capture conceptual similarity; BM25 captures precise financial metric acronyms (*"ARR"*, *"EBITDA"*, *"CapEx"*). RRF combines both without scale distortion.
                """
            )

with col1:
    with st.container(border=True):
        st.markdown("### 3. FlashRank Cross-Encoder Reranking")
        st.markdown(
            """
            - **Model:** `ms-marco-MiniLM-L-12-v2` running via ONNX runtime on CPU.
            - **Latency:** Fast CPU execution (~4.2s for top 25 candidates).
            - **Deep Attention:** Unlike bi-encoders, the cross-encoder computes full token-to-token cross-attention between user query and passage tokens.
            - **Benchmark Impact:** Boosted exact page Mean Reciprocal Rank (MRR@5) from **0.2453 to 0.3088 (+25.9%)**.
            """
        )

with col2:
    with st.container(border=True):
        st.markdown("### 4. Abstention Guardrail & LLM Resiliency")
        st.markdown(
            """
            - **Calibrated Abstention:** If the highest reranker confidence is below **0.30**, the query is flagged as out-of-corpus.
            - **Instant Rejection:** Prevents the LLM from making speculative guesses or financial hallucinations, and saves token consumption.
            - **Multi-Provider Fallback:**
              - Primary: Google Gemini 2.5 Flash (`gemini-2.5-flash`).
              - Fallback: Groq Cloud Llama 3.3 70B (`llama-3.3-70b-versatile`).
              - Automatic failover seamlessly engages on HTTP 429 quota exhaustion.
            """
        )
