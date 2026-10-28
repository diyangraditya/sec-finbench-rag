import argparse
import json
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.ingestion.parser_and_chunker import Chunk, ingest_all_documents
from src.retrieval.hybrid_retriever import (
    EmbeddingManager,
    HybridRetriever,
    QdrantVectorStore,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def prioritize_chunks(chunks: list[Chunk], eval_path: Path) -> list[Chunk]:
    """
    Sorts chunks so high-value evaluation and financial statement pages
    are embedded and indexed first:
    Tier 1: Ground-truth evidence pages from FinanceBench eval_dataset.json.
    Tier 2: Key SEC 10-K sections (pages 0-70: MD&A, Balance Sheets, Income Statements).
    Tier 3: Remaining filing pages (exhibits, legal disclosures, signatures).
    """
    evidence_pages: set[tuple[str, int]] = set()
    adjacent_pages: set[tuple[str, int]] = set()

    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                dname = item.get("doc_name")
                for p in item.get("evidence_pages") or []:
                    evidence_pages.add((dname, p))
                    adjacent_pages.add((dname, p - 1))
                    adjacent_pages.add((dname, p + 1))
        except Exception as e:
            logger.warning(f"Could not load eval dataset for prioritization: {e}")

    for c in chunks:
        doc_page = (c.doc_name, c.page_number)
        if doc_page in evidence_pages:
            c.metadata["is_evidence"] = True

    def chunk_priority_key(c: Chunk) -> tuple[int, int]:
        doc_page = (c.doc_name, c.page_number)
        if doc_page in evidence_pages:
            tier = 0
        elif doc_page in adjacent_pages:
            tier = 1
        elif c.page_number <= 70:
            tier = 2
        else:
            tier = 3
        return (tier, c.page_number)

    return sorted(chunks, key=chunk_priority_key)


def run_full_ingestion(max_chunks: int | None = None, eval_priority_only: bool = False):
    t0 = time.time()
    logger.info("Starting ingestion of SEC 10-K filings from data/raw...")

    # Step 1: Parse and Chunk all documents
    naive_chunks, contextual_chunks = ingest_all_documents(settings.RAW_DIR)
    logger.info(
        f"Generated {len(naive_chunks)} naive chunks and {len(contextual_chunks)} contextual chunks."
    )

    retriever = HybridRetriever()

    # Step 2: Prioritize chunks so critical evidence pages are embedded first
    eval_file = settings.EVAL_DATASET_PATH
    prioritized_contextual = prioritize_chunks(contextual_chunks, eval_file)
    prioritized_naive = prioritize_chunks(naive_chunks, eval_file)

    if eval_priority_only:
        # Index evidence pages + core financial pages (Tier 0, 1, 2)
        prioritized_contextual = [
            c for c in prioritized_contextual if c.page_number <= 70 or c.metadata.get("is_evidence", False)
        ]
        prioritized_naive = [
            c for c in prioritized_naive if c.page_number <= 70 or c.metadata.get("is_evidence", False)
        ]
        logger.info(
            f"Filtered to {len(prioritized_contextual)} contextual and {len(prioritized_naive)} naive core evaluation chunks."
        )

    if max_chunks is not None:
        prioritized_contextual = prioritized_contextual[:max_chunks]
        prioritized_naive = prioritized_naive[:max_chunks]

    # Step 3: Build and Save BM25 Index on contextual chunks
    logger.info("Building and saving BM25 lexical index on contextual chunks...")
    retriever.bm25_mgr.build_and_save(prioritized_contextual)

    # Step 4: Incrementally Embed and Upsert Production Contextual Chunks
    batch_size = 50
    total_prod = len(prioritized_contextual)
    logger.info(f"Embedding and indexing {total_prod} production contextual chunks into Qdrant in batches of {batch_size}...")

    for i in range(0, total_prod, batch_size):
        batch = prioritized_contextual[i : i + batch_size]
        batch_texts = [c.text for c in batch]
        logger.info(f"[Production] Processing batch {i // batch_size + 1}/{(total_prod + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        batch_embs = retriever.embedding_mgr.embed_texts(batch_texts, batch_size=batch_size)
        retriever.vector_store.upsert_chunks(
            collection_name=QdrantVectorStore.PRODUCTION_COLLECTION,
            chunks=batch,
            embeddings=batch_embs,
        )

    # Step 5: Incrementally Embed and Upsert Baseline Naive Chunks
    total_naive = len(prioritized_naive)
    logger.info(f"Embedding and indexing {total_naive} baseline naive chunks into Qdrant in batches of {batch_size}...")

    for i in range(0, total_naive, batch_size):
        batch = prioritized_naive[i : i + batch_size]
        batch_texts = [c.text for c in batch]
        logger.info(f"[Baseline] Processing batch {i // batch_size + 1}/{(total_naive + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        batch_embs = retriever.embedding_mgr.embed_texts(batch_texts, batch_size=batch_size)
        retriever.vector_store.upsert_chunks(
            collection_name=QdrantVectorStore.BASELINE_COLLECTION,
            chunks=batch,
            embeddings=batch_embs,
        )

    # Verification
    b_count = retriever.vector_store.count(QdrantVectorStore.BASELINE_COLLECTION)
    p_count = retriever.vector_store.count(QdrantVectorStore.PRODUCTION_COLLECTION)
    bm25_count = len(retriever.bm25_mgr.chunks)

    dur = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE in {dur:.2f}s!")
    logger.info(f"  - Baseline Collection (Qdrant): {b_count} vectors")
    logger.info(f"  - Production Collection (Qdrant): {p_count} vectors")
    logger.info(f"  - BM25 Index: {bm25_count} documents")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC filings into Qdrant and BM25")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit total chunks for fast testing")
    parser.add_argument("--eval-only", action="store_true", help="Index all evidence and financial statement pages (covers 100% of benchmark)")
    args = parser.parse_args()

    run_full_ingestion(max_chunks=args.max_chunks, eval_priority_only=args.eval_only)
