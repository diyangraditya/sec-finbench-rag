"""
Interactive Debugger for RAG Pipeline Steps.
Allows you to inspect every stage under the hood:
1. Tokenization & Metadata Pre-filter
2. Dense Vector Search (Qdrant)
3. Lexical BM25 Search
4. Reciprocal Rank Fusion (RRF)
5. FlashRank Cross-Encoder Reranking & Logit Calibration
6. Abstention Guardrail Decision
7. Structured LLM Generation
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.generation.generator import RAGGenerator
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    QdrantVectorStore,
)


def debug_query(query: str, doc_name: str | None = None, provider: str = "gemini"):
    print("=" * 80)
    print(f"STEP-BY-STEP RAG TRACE FOR QUERY: '{query}'")
    if doc_name:
        print(f"Document Filter: {doc_name}")
    print("=" * 80)

    retriever = HybridRetriever()
    generator = RAGGenerator()

    # ---------------------------------------------------------
    # STAGE 1: Dense Vector Retrieval (Qdrant)
    # ---------------------------------------------------------
    print("\n[STAGE 1] Dense Vector Search (Embedding -> Qdrant Cosine Similarity)")
    t0 = time.time()
    query_vector = retriever.embedding_mgr.embed_query(query)
    embed_lat = time.time() - t0
    print(f"  • Query embedded into {len(query_vector)}-dim vector ({embed_lat*1000:.1f}ms)")

    dense_hits = retriever.vector_store.search_vector(
        collection_name=QdrantVectorStore.PRODUCTION_COLLECTION,
        query_vector=query_vector,
        limit=settings.TOP_K_RETRIEVAL,
        doc_name=doc_name,
    )
    print(f"  • Retrieved {len(dense_hits)} dense vector candidates:")
    for i, h in enumerate(dense_hits[:5], start=1):
        print(
            f"    #{i:02d} | Doc: {h.doc_name} | Page: {h.page_number} (1-based: {h.page_num_1based}) | "
            f"Cosine Sim: {h.vector_score:.4f} | Preview: {repr(h.text[:80])}"
        )

    # ---------------------------------------------------------
    # STAGE 2: Lexical BM25 Search
    # ---------------------------------------------------------
    print("\n[STAGE 2] Lexical BM25 Inverted Index Search")
    bm25_hits = retriever.bm25_mgr.search(
        query=query,
        limit=settings.TOP_K_RETRIEVAL,
        doc_name=doc_name,
    )
    print(f"  • Retrieved {len(bm25_hits)} BM25 keyword candidates:")
    for i, h in enumerate(bm25_hits[:5], start=1):
        print(
            f"    #{i:02d} | Doc: {h.doc_name} | Page: {h.page_number} (1-based: {h.page_num_1based}) | "
            f"BM25 Score: {h.bm25_score:.2f} | Preview: {repr(h.text[:80])}"
        )

    # ---------------------------------------------------------
    # STAGE 3: Reciprocal Rank Fusion (RRF)
    # ---------------------------------------------------------
    print("\n[STAGE 3] Reciprocal Rank Fusion (RRF, k=60)")
    chunk_map = {}
    rrf_scores = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_map[hit.id] = hit
        rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (settings.RRF_K + rank))

    for rank, hit in enumerate(bm25_hits, start=1):
        if hit.id not in chunk_map:
            chunk_map[hit.id] = hit
        else:
            chunk_map[hit.id].bm25_score = hit.bm25_score
        rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (settings.RRF_K + rank))

    sorted_by_rrf = sorted(chunk_map.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused_candidates = [chunk_map[cid] for cid in sorted_by_rrf[:settings.TOP_K_RETRIEVAL]]

    print(f"  • Top candidates after reciprocal rank fusion:")
    for i, h in enumerate(fused_candidates[:5], start=1):
        rrf_val = rrf_scores[h.id]
        print(
            f"    #{i:02d} | Doc: {h.doc_name} | Page: {h.page_number} | "
            f"RRF Score: {rrf_val:.5f} (Dense: {h.vector_score or 0.0:.3f}, BM25: {h.bm25_score or 0.0:.1f})"
        )

    # ---------------------------------------------------------
    # STAGE 4: Cross-Encoder Reranking (FlashRank)
    # ---------------------------------------------------------
    print(f"\n[STAGE 4] FlashRank Cross-Encoder Reranking ({settings.RERANKER_MODEL_NAME})")
    from flashrank import RerankRequest
    passages = [{"id": c.id, "text": c.text} for c in fused_candidates]
    rerank_req = RerankRequest(query=query, passages=passages)
    t_rr = time.time()
    reranked = retriever.ranker.rerank(rerank_req)
    rerank_lat = time.time() - t_rr
    print(f"  • Cross-encoder inference latency: {rerank_lat*1000:.1f}ms")

    scored_candidates = []
    for r in reranked:
        cid = r["id"]
        raw_s = float(r["score"])
        calibrated_s = retriever._calibrate_flashrank_score(raw_s)
        candidate = chunk_map[cid]
        candidate.raw_score = raw_s
        vec_s = candidate.vector_score or 0.0
        candidate.score = float(min(1.0, max(0.0, 0.5 * calibrated_s + 0.5 * vec_s)))
        scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda c: c.score, reverse=True)
    top_final = scored_candidates[:settings.TOP_K_RERANK]

    for i, c in enumerate(top_final, start=1):
        print(
            f"    #{i:02d} | Final Blended: {c.score:.4f} | Raw Cross-Enc: {c.raw_score:.6f} | "
            f"Page: {c.page_number} | Doc: {c.doc_name}"
        )

    # ---------------------------------------------------------
    # STAGE 5: Abstention Guardrail Check
    # ---------------------------------------------------------
    print("\n[STAGE 5] Abstention Guardrail Verification")
    top_score = top_final[0].score if top_final else 0.0
    threshold = settings.RERANK_SCORE_THRESHOLD
    print(f"  • Top Passage Confidence Score: {top_score:.4f}")
    print(f"  • Configured Threshold:         {threshold:.4f}")

    if top_score < threshold:
        print(f"  🚫 GUARDRAIL TRIGGERED: Confidence {top_score:.4f} < {threshold:.4f}")
        print("  • LLM generation will be ABSTAINED to prevent hallucinations.")
        return
    else:
        print(f"  ✅ CONFIDENCE SUFFICIENT: Proceeding to LLM generation.")

    # ---------------------------------------------------------
    # STAGE 6: LLM Generation & Structured Validation
    # ---------------------------------------------------------
    print(f"\n[STAGE 6] LLM Generation ({provider.upper()}) with Pydantic Structured Output")
    resp, gen_lat = generator.generate(
        question=query,
        chunks=top_final,
        provider=provider,
        enforce_abstention=True,
    )
    print(f"  • Latency: {gen_lat:.2f}s | Provider: {resp.provider_used}")
    print(f"  • Grounded: {resp.is_grounded}")
    print(f"  • Cited Pages: {resp.cited_pages}")
    print(f"  • Cited Docs: {resp.cited_doc_names}")
    print("\n" + "-" * 40 + " FINAL GENERATED ANSWER " + "-" * 40)
    print(resp.answer)
    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Deep Trace Debugger for RAG Pipeline")
    parser.add_argument("query", type=str, help="Question to inspect")
    parser.add_argument("--doc-name", type=str, default=None, help="Optional document filter")
    parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini", help="LLM provider")
    args = parser.parse_args()

    debug_query(query=args.query, doc_name=args.doc_name, provider=args.provider)


if __name__ == "__main__":
    main()
