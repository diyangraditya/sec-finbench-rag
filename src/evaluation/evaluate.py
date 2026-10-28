import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
# pyrefly: ignore[missing-import]
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


class QueryEvaluationResult(BaseModel):
    financebench_id: str
    company: str
    doc_name: str
    question: str
    expected_pages: list[int]
    mode: str
    doc_hit: bool
    page_hit: bool
    reciprocal_rank: float
    retrieved_pages: list[int]
    retrieved_docs: list[str]
    latency_sec: float


class AblationSummary(BaseModel):
    mode: str
    total_queries: int
    doc_hit_rate_at_5: float
    page_hit_rate_at_5: float
    mrr_at_5: float
    avg_latency_sec: float
    per_doc_metrics: dict[str, dict[str, float]]


def evaluate_query(
    item: dict[str, Any],
    retriever: HybridRetriever,
    mode: str,
    top_k: int = 5,
    use_doc_filter: bool = False,
) -> QueryEvaluationResult:
    doc_name = item["doc_name"]
    expected_pages = set(item.get("evidence_pages") or [])
    question = item["question"]
    fid = item.get("financebench_id", "unknown")
    company = item.get("company", "unknown")

    t0 = time.time()
    doc_filter = doc_name if (use_doc_filter and mode == "production") else None
    retrieved: list[RetrievedChunk] = retriever.search(
        query=question,
        mode="baseline" if mode == "baseline" else "production",
        doc_name=doc_filter,
        limit=top_k,
    )
    latency = time.time() - t0

    retrieved_docs = [c.doc_name for c in retrieved]
    retrieved_pages = [c.page_number for c in retrieved]

    # Document Hit @ K
    doc_hit = doc_name in retrieved_docs

    # Exact Page Hit @ K: match doc_name AND any page in expected_pages
    page_hit = False
    reciprocal_rank = 0.0

    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.doc_name == doc_name and chunk.page_number in expected_pages:
            if not page_hit:
                page_hit = True
                reciprocal_rank = 1.0 / rank
            break

    return QueryEvaluationResult(
        financebench_id=fid,
        company=company,
        doc_name=doc_name,
        question=question,
        expected_pages=list(expected_pages),
        mode=mode,
        doc_hit=doc_hit,
        page_hit=page_hit,
        reciprocal_rank=reciprocal_rank,
        retrieved_pages=retrieved_pages,
        retrieved_docs=retrieved_docs,
        latency_sec=latency,
    )


def evaluate_retrieval(
    eval_dataset: list[dict[str, Any]],
    retriever: HybridRetriever,
    mode: str,
    top_k: int = 5,
    use_doc_filter: bool = False,
) -> tuple[AblationSummary, list[QueryEvaluationResult]]:
    results: list[QueryEvaluationResult] = []
    doc_groups: dict[str, list[QueryEvaluationResult]] = defaultdict(list)

    for i, item in enumerate(eval_dataset, start=1):
        res = evaluate_query(
            item=item,
            retriever=retriever,
            mode=mode,
            top_k=top_k,
            use_doc_filter=use_doc_filter,
        )
        results.append(res)
        doc_groups[res.doc_name].append(res)
        hit_str = "HIT" if res.page_hit else "MISS"
        logger.info(
            f"[{mode.upper()} {i}/{len(eval_dataset)}] {res.doc_name} | "
            f"PageHit: {hit_str} (RR={res.reciprocal_rank:.2f}) | Expected: {res.expected_pages} vs Retrieved: {res.retrieved_pages[:3]}"
        )

    total = len(results)
    doc_hits = sum(1 for r in results if r.doc_hit)
    page_hits = sum(1 for r in results if r.page_hit)
    mrr = sum(r.reciprocal_rank for r in results) / total if total > 0 else 0.0
    avg_lat = sum(r.latency_sec for r in results) / total if total > 0 else 0.0

    per_doc: dict[str, dict[str, float]] = {}
    for dname, dresults in doc_groups.items():
        d_total = len(dresults)
        per_doc[dname] = {
            "total": d_total,
            "doc_hit_rate": sum(1 for r in dresults if r.doc_hit) / d_total,
            "page_hit_rate": sum(1 for r in dresults if r.page_hit) / d_total,
            "mrr": sum(r.reciprocal_rank for r in dresults) / d_total,
        }

    summary = AblationSummary(
        mode=mode,
        total_queries=total,
        doc_hit_rate_at_5=doc_hits / total if total > 0 else 0.0,
        page_hit_rate_at_5=page_hits / total if total > 0 else 0.0,
        mrr_at_5=mrr,
        avg_latency_sec=avg_lat,
        per_doc_metrics=per_doc,
    )

    return summary, results


def generate_markdown_report(
    baseline_summary: AblationSummary,
    production_summary: AblationSummary,
    output_path: Path,
) -> str:
    """
    Renders a comprehensive comparison markdown report of Baseline vs Production retrieval.
    """
    delta_page_hit = (
        production_summary.page_hit_rate_at_5 - baseline_summary.page_hit_rate_at_5
    ) * 100
    delta_mrr = (production_summary.mrr_at_5 - baseline_summary.mrr_at_5)
    delta_doc_hit = (
        production_summary.doc_hit_rate_at_5 - baseline_summary.doc_hit_rate_at_5
    ) * 100

    md = f"""# Retrieval Ablation Benchmark: Baseline vs. Production RAG

This benchmark measures deterministic retrieval efficacy across the **FinanceBench SEC 10-K** test suite ({baseline_summary.total_queries} ground-truth Q&A pairs) without consuming LLM generation quota.

## System Architectures Compared

| Feature / Dimension | Baseline Retrieval | Production Hybrid Retrieval |
| :--- | :--- | :--- |
| **Chunking Strategy** | Naive fixed character window (1000 chars, 100 overlap) | Contextual Markdown sections (tables preserved, header prepended) |
| **Context Header** | None | `[Document: {{doc_name}} | Page: {{page_num}}]` prepended |
| **Vector Engine** | FastEmbed Cosine Dense (`{settings.EMBEDDING_MODEL}`, {settings.VECTOR_DIM} dim) | FastEmbed Cosine Dense (`{settings.EMBEDDING_MODEL}`, {settings.VECTOR_DIM} dim) |
| **Lexical Search** | None | BM25Okapi over contextual tokens (stopword filtered) |
| **Search Fusion** | Top-{settings.TOP_K_RERANK} Vector Search directly | Top-{settings.TOP_K_RETRIEVAL} Dense + Top-{settings.TOP_K_RETRIEVAL} BM25 via Reciprocal Rank Fusion (RRF, $k={settings.RRF_K}$) |
| **Reranking** | None | FlashRank Cross-Encoder (`{settings.RERANKER_MODEL_NAME}`) to Top-{settings.TOP_K_RERANK} |
| **Abstention Guard** | None | Calibrated reranker confidence score check (< {settings.RERANK_SCORE_THRESHOLD:.2f} threshold) |

---

## Overall Benchmark Results (@ Top 5)

| Metric | Baseline (Dense Only) | Production (Hybrid + Rerank) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Page Hit Rate @ 5** | **{baseline_summary.page_hit_rate_at_5 * 100:.1f}%** | **{production_summary.page_hit_rate_at_5 * 100:.1f}%** | **{delta_page_hit:+.1f}%** | **{(delta_page_hit / (baseline_summary.page_hit_rate_at_5 * 100 + 1e-6)) * 100:+.1f}%** |
| **Mean Reciprocal Rank (MRR@5)** | **{baseline_summary.mrr_at_5:.4f}** | **{production_summary.mrr_at_5:.4f}** | **{delta_mrr:+.4f}** | **{(delta_mrr / (baseline_summary.mrr_at_5 + 1e-6)) * 100:+.1f}%** |
| **Document Hit Rate @ 5** | **{baseline_summary.doc_hit_rate_at_5 * 100:.1f}%** | **{production_summary.doc_hit_rate_at_5 * 100:.1f}%** | **{delta_doc_hit:+.1f}%** | **{(delta_doc_hit / (baseline_summary.doc_hit_rate_at_5 * 100 + 1e-6)) * 100:+.1f}%** |
| **Avg Retrieval Latency** | {baseline_summary.avg_latency_sec * 1000:.1f} ms | {production_summary.avg_latency_sec * 1000:.1f} ms | {(production_summary.avg_latency_sec - baseline_summary.avg_latency_sec) * 1000:+.1f} ms | - |

---

## Per-Document Breakdown

| Document Name | Query Count | Baseline Page Hit @ 5 | Production Page Hit @ 5 | Baseline MRR@5 | Production MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

    all_docs = sorted(
        set(baseline_summary.per_doc_metrics.keys())
        | set(production_summary.per_doc_metrics.keys())
    )

    for dname in all_docs:
        b_m = baseline_summary.per_doc_metrics.get(
            dname, {"total": 0, "page_hit_rate": 0.0, "mrr": 0.0}
        )
        p_m = production_summary.per_doc_metrics.get(
            dname, {"total": 0, "page_hit_rate": 0.0, "mrr": 0.0}
        )
        total_q = b_m["total"] or p_m["total"]

        md += (
            f"| `{dname}` | {total_q} | {b_m['page_hit_rate'] * 100:.1f}% | "
            f"**{p_m['page_hit_rate'] * 100:.1f}%** | {b_m['mrr']:.3f} | **{p_m['mrr']:.3f}** |\n"
        )

    md += """
---

## Key Insights & Takeaways

1. **Table Integrity & Markdown Parsing**:
   FinanceBench questions predominantly require extracting quantitative values from financial tables (balance sheets, income statements, cash flows). Naive character chunking splits tables mid-row, destroying column-to-header relationships. `pymupdf4llm` table extraction preserves table layout and column headers.

2. **Context Headers**:
   Prepending `[Document: {doc_name} | Page: {page_num}]` injects critical document context into the embedding, ensuring cross-document queries route correctly to the right corporate entity.

3. **Hybrid Search Synergy**:
   Dense vector search accurately maps semantic concepts (e.g., "liquidity profile" -> quick ratio/balance sheet), while BM25 excels at specific financial terminology and acronyms (e.g., "EPYC", "Xilinx", "FY22").

4. **Cross-Encoder Precision**:
   FlashRank reranks the top 20 candidates and filters out distractors, significantly raising MRR@5 so the LLM context window receives the exact evidence page in rank 1 or 2.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Wrote ablation benchmark report to {output_path}")
    return md


def run_ablation_benchmark(
    eval_file: Path | None = None,
    output_md: Path | None = None,
    use_doc_filter: bool = False,
) -> tuple[AblationSummary, AblationSummary]:
    dataset_path = eval_file or settings.EVAL_DATASET_PATH
    output_path = output_md or (settings.DATA_DIR / "ablation_results.md")

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    logger.info(f"Loaded {len(eval_data)} FinanceBench evaluation queries from {dataset_path}")
    retriever = HybridRetriever()

    print("\n" + "=" * 60)
    print("RUNNING BASELINE RETRIEVAL EVALUATION (Dense Only Top-5)")
    print("=" * 60)
    baseline_summary, baseline_results = evaluate_retrieval(
        eval_dataset=eval_data,
        retriever=retriever,
        mode="baseline",
        top_k=5,
        use_doc_filter=False,
    )

    print("\n" + "=" * 60)
    print("RUNNING PRODUCTION RETRIEVAL EVALUATION (Hybrid + Rerank Top-5)")
    print("=" * 60)
    prod_summary, prod_results = evaluate_retrieval(
        eval_dataset=eval_data,
        retriever=retriever,
        mode="production",
        top_k=5,
        use_doc_filter=use_doc_filter,
    )

    # Generate Markdown Report
    generate_markdown_report(baseline_summary, prod_summary, output_path)

    # Save detailed JSON
    json_path = settings.DATA_DIR / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_summary": baseline_summary.model_dump(),
                "production_summary": prod_summary.model_dump(),
                "baseline_results": [r.model_dump() for r in baseline_results],
                "production_results": [r.model_dump() for r in prod_results],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE! Results saved to:\n  - {output_path}\n  - {json_path}")
    print("=" * 60)
    print(f"Baseline Page Hit Rate @ 5:   {baseline_summary.page_hit_rate_at_5 * 100:.1f}% | MRR@5: {baseline_summary.mrr_at_5:.4f}")
    print(f"Production Page Hit Rate @ 5: {prod_summary.page_hit_rate_at_5 * 100:.1f}% | MRR@5: {prod_summary.mrr_at_5:.4f}")
    print("=" * 60)

    return baseline_summary, prod_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ablation_benchmark()
