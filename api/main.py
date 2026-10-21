import logging
import time
from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.generation.generator import RAGGenerator, RAGResponse
from src.ingestion.parser_and_chunker import (
    ingest_all_documents,
    parse_pdf_to_pages,
    naive_chunk,
    contextual_markdown_chunk,
)
from src.retrieval.hybrid_retriever import (
    EmbeddingManager,
    HybridRetriever,
    QdrantVectorStore,
    RetrievedChunk,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="FinanceBench Production RAG API",
    description="High-performance SEC 10-K Retrieval-Augmented Generation with Hybrid Search and LLM fallback.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons
retriever = HybridRetriever()
generator = RAGGenerator()


class QueryRequest(BaseModel):
    question: str = Field(..., description="Financial question regarding the SEC 10-K filings.")
    mode: Literal["baseline", "production"] = Field(
        default="production",
        description="Retrieval mode: 'baseline' (vector only) or 'production' (hybrid + rerank)."
    )
    provider: Literal["gemini", "groq"] | None = Field(
        default=None,
        description="LLM provider override ('gemini' or 'groq'). Defaults to config."
    )
    doc_name: str | None = Field(
        default=None,
        description="Optional metadata pre-filter targeting a specific SEC 10-K document."
    )
    top_k: int = Field(default=5, description="Number of passages to pass to generation.")


class QueryResponse(BaseModel):
    question: str
    mode: str
    answer: str
    cited_pages: list[int]
    cited_doc_names: list[str]
    is_grounded: bool
    provider_used: str
    total_latency_sec: float
    retrieval_latency_sec: float
    generation_latency_sec: float
    retrieved_chunks: list[RetrievedChunk]


class IngestResponse(BaseModel):
    status: str
    naive_chunks_count: int
    contextual_chunks_count: int
    baseline_qdrant_count: int
    production_qdrant_count: int
    duration_sec: float


class CompareRequest(BaseModel):
    question: str
    provider: Literal["gemini", "groq"] | None = None
    doc_name: str | None = None


class CompareResponse(BaseModel):
    question: str
    baseline: QueryResponse
    production: QueryResponse


@app.get("/health")
def health_check() -> dict[str, Any]:
    b_count = retriever.vector_store.count(QdrantVectorStore.BASELINE_COLLECTION)
    p_count = retriever.vector_store.count(QdrantVectorStore.PRODUCTION_COLLECTION)
    bm25_count = len(retriever.bm25_mgr.chunks)

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "config": {
            "llm_provider": settings.LLM_PROVIDER,
            "gemini_model": settings.GEMINI_MODEL,
            "groq_model": settings.GROQ_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK,
            "rerank_score_threshold": settings.RERANK_SCORE_THRESHOLD,
        },
        "storage": {
            "baseline_chunks_in_qdrant": b_count,
            "production_chunks_in_qdrant": p_count,
            "bm25_indexed_chunks": bm25_count,
            "qdrant_path": settings.QDRANT_STORAGE_PATH,
        },
        "keys_configured": {
            "gemini": bool(settings.GEMINI_API_KEY),
            "groq": bool(settings.GROQ_API_KEY),
        },
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents() -> IngestResponse:
    t0 = time.time()
    try:
        # Ingest all raw PDFs
        naive_chunks, contextual_chunks = ingest_all_documents(settings.RAW_DIR)

        if not naive_chunks and not contextual_chunks:
            raise HTTPException(status_code=400, detail="No PDF documents found in data/raw")

        # 1. Embed and index baseline naive chunks
        logger.info(f"Embedding {len(naive_chunks)} naive chunks for baseline...")
        naive_texts = [c.text for c in naive_chunks]
        naive_embs = retriever.embedding_mgr.embed_texts(naive_texts)
        retriever.vector_store.upsert_chunks(
            collection_name=QdrantVectorStore.BASELINE_COLLECTION,
            chunks=naive_chunks,
            embeddings=naive_embs,
        )

        # 2. Embed and index production contextual chunks
        logger.info(f"Embedding {len(contextual_chunks)} contextual chunks for production...")
        contextual_texts = [c.text for c in contextual_chunks]
        contextual_embs = retriever.embedding_mgr.embed_texts(contextual_texts)
        retriever.vector_store.upsert_chunks(
            collection_name=QdrantVectorStore.PRODUCTION_COLLECTION,
            chunks=contextual_chunks,
            embeddings=contextual_embs,
        )

        # 3. Build BM25 index on contextual chunks
        logger.info(f"Building BM25 index over {len(contextual_chunks)} chunks...")
        retriever.bm25_mgr.build_and_save(contextual_chunks)

        b_count = retriever.vector_store.count(QdrantVectorStore.BASELINE_COLLECTION)
        p_count = retriever.vector_store.count(QdrantVectorStore.PRODUCTION_COLLECTION)
        dur = time.time() - t0

        return IngestResponse(
            status="success",
            naive_chunks_count=len(naive_chunks),
            contextual_chunks_count=len(contextual_chunks),
            baseline_qdrant_count=b_count,
            production_qdrant_count=p_count,
            duration_sec=dur,
        )
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest) -> QueryResponse:
    t0 = time.time()

    # Step 1: Retrieval
    t_ret_start = time.time()
    chunks = retriever.search(
        query=req.question,
        mode=req.mode,
        doc_name=req.doc_name,
        limit=req.top_k,
    )
    retrieval_lat = time.time() - t_ret_start

    # Step 2: Generation (with abstention guardrail for production mode)
    enforce_abstention = req.mode == "production"
    rag_resp, gen_lat = generator.generate(
        question=req.question,
        chunks=chunks,
        provider=req.provider,
        enforce_abstention=enforce_abstention,
    )

    total_lat = time.time() - t0

    return QueryResponse(
        question=req.question,
        mode=req.mode,
        answer=rag_resp.answer,
        cited_pages=rag_resp.cited_pages,
        cited_doc_names=rag_resp.cited_doc_names,
        is_grounded=rag_resp.is_grounded,
        provider_used=rag_resp.provider_used,
        total_latency_sec=total_lat,
        retrieval_latency_sec=retrieval_lat,
        generation_latency_sec=gen_lat,
        retrieved_chunks=chunks,
    )


@app.post("/query/compare", response_model=CompareResponse)
def compare_query(req: CompareRequest) -> CompareResponse:
    """Executes both baseline and production RAG on the same query side-by-side."""
    base_req = QueryRequest(
        question=req.question,
        mode="baseline",
        provider=req.provider,
        doc_name=req.doc_name,
    )
    prod_req = QueryRequest(
        question=req.question,
        mode="production",
        provider=req.provider,
        doc_name=req.doc_name,
    )

    base_resp = execute_query(base_req)
    prod_resp = execute_query(prod_req)

    return CompareResponse(
        question=req.question,
        baseline=base_resp,
        production=prod_resp,
    )
