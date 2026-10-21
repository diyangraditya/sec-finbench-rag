import hashlib
import json
import logging
import math
import pickle
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Literal
import numpy as np
from flashrank import Ranker, RerankRequest
from google import genai
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.parser_and_chunker import Chunk

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    id: str
    doc_name: str
    page_number: int  # 0-indexed
    page_num_1based: int  # 1-indexed
    chunk_index: int
    chunk_mode: str
    text: str
    raw_content: str
    score: float  # Final relevance / rerank score
    raw_score: float | None = None
    rrf_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rank: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingManager:
    """
    Manages embedding generation via Gemini API with disk-backed SQLite caching
    and clean batching to respect free-tier rate limits.
    """

    def __init__(self, db_path: Path | None = None, model: str | None = None):
        self.db_path = db_path or settings.EMBEDDING_CACHE_DB
        self.model = model or settings.EMBEDDING_MODEL
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)")
            conn.commit()

    @staticmethod
    def _hash_text(text: str, model: str) -> str:
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_cached_embedding(self, text: str) -> list[float] | None:
        h = self._hash_text(text, self.model)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE hash = ?", (h,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def get_cached_embeddings_bulk(self, texts: list[str]) -> dict[int, list[float]]:
        """Returns dict of {index_in_texts: embedding} for cached items across compatible 3072-dim models."""
        results: dict[int, list[float]] = {}
        # Candidate models to search in cache
        candidate_models = [self.model, "gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview"]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, t in enumerate(texts):
                for m in candidate_models:
                    h = self._hash_text(t, m)
                    cursor.execute("SELECT embedding FROM embeddings WHERE hash = ?", (h,))
                    row = cursor.fetchone()
                    if row:
                        results[i] = json.loads(row[0])
                        break

        return results

    def save_embeddings_bulk(self, text_emb_pairs: list[tuple[str, list[float]]]) -> None:
        records = [
            (
                self._hash_text(t, self.model),
                self.model,
                json.dumps(emb),
            )
            for t, emb in text_emb_pairs
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (hash, model, embedding) VALUES (?, ?, ?)",
                records,
            )
            conn.commit()

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Embeds a list of texts with disk caching, cross-model reuse, and rate-limit aware batching.
        """
        if not texts:
            return []

        # 1. Check disk cache first (checks all 3072-dim models)
        cached = self.get_cached_embeddings_bulk(texts)
        missing_indices = [i for i in range(len(texts)) if i not in cached]

        if not missing_indices:
            return [cached[i] for i in range(len(texts))]

        logger.info(
            f"Embedding {len(texts)} texts: {len(cached)} cached, {len(missing_indices)} to fetch via API"
        )

        client = self._get_client()

        # 2. Batch requests for uncached texts
        for i in range(0, len(missing_indices), batch_size):
            batch_idxs = missing_indices[i : i + batch_size]
            batch_texts = [texts[idx] for idx in batch_idxs]

            # Retry loop with model fallback and exponential backoff
            max_retries = 6
            embeddings = None
            models_to_try = [self.model] + [
                m for m in getattr(settings, "EMBEDDING_FALLBACK_MODELS", []) if m != self.model
            ]

            for active_model in models_to_try:
                self.model = active_model
                for attempt in range(max_retries):
                    try:
                        response = client.models.embed_content(
                            model=self.model,
                            contents=batch_texts,
                        )
                        embeddings = [e.values for e in response.embeddings]
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        # If daily quota exceeded on this model, switch to fallback model immediately
                        if "perday" in err_str or "daily" in err_str:
                            logger.warning(
                                f"Embedding model {self.model} hit daily quota, switching to next fallback model..."
                            )
                            break

                        if "429" in err_str or "quota" in err_str or "unavailable" in err_str or "503" in err_str or "resource_exhausted" in err_str:
                            wait_match = re.search(r"retry in ([\d\.]+)s", err_str)
                            if wait_match:
                                wait_sec = float(wait_match.group(1)) + 2.0
                            else:
                                wait_sec = max(30.0, float((2**attempt) * 5))

                            logger.warning(
                                f"Embedding API quota/rate-limit on {self.model} (attempt {attempt + 1}/{max_retries}), pausing {wait_sec:.1f}s: {e}"
                            )
                            time.sleep(wait_sec)
                        else:
                            logger.error(f"Embedding error on {self.model}: {e}")
                            raise e

                if embeddings is not None:
                    break

            # Save newly fetched embeddings to disk cache immediately
            pairs_to_save = list(zip(batch_texts, embeddings))
            self.save_embeddings_bulk(pairs_to_save)

            for idx, emb in zip(batch_idxs, embeddings):
                cached[idx] = emb

            # Pacing to stay comfortably within free tier limits (~15 RPM)
            if i + batch_size < len(missing_indices):
                time.sleep(2.0)

        return [cached[i] for i in range(len(texts))]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class QdrantVectorStore:
    """
    Local Qdrant vector store storing baseline and production collections.
    Releases storage file lock after each operation for concurrency.
    """

    BASELINE_COLLECTION = "baseline_chunks"
    PRODUCTION_COLLECTION = "production_chunks"
    VECTOR_DIM = 3072

    def __init__(self, path: str | None = None):
        self.path = path or settings.QDRANT_STORAGE_PATH
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self._ensure_collections()

    def _get_client(self) -> QdrantClient:
        return QdrantClient(path=self.path)

    def _ensure_collections(self) -> None:
        try:
            client = self._get_client()
            for collection_name in [self.BASELINE_COLLECTION, self.PRODUCTION_COLLECTION]:
                if not client.collection_exists(collection_name):
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qmodels.VectorParams(
                            size=self.VECTOR_DIM,
                            distance=qmodels.Distance.COSINE,
                        ),
                    )
                    logger.info(f"Created Qdrant collection: {collection_name}")
            client.close()
        except Exception as e:
            logger.debug(f"Collection check: {e}")

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        if not chunks or not embeddings:
            return

        points: list[qmodels.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.id,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "page_num_1based": chunk.page_num_1based,
                "chunk_index": chunk.chunk_index,
                "chunk_mode": chunk.chunk_mode,
                "text": chunk.text,
                "raw_content": chunk.raw_content,
                "metadata": chunk.metadata,
            }
            points.append(
                qmodels.PointStruct(
                    id=self._chunk_id_to_uuid(chunk.id),
                    vector=emb,
                    payload=payload,
                )
            )

        client = self._get_client()
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"Upserted {len(points)} points into {collection_name}")
        finally:
            client.close()

    def count(self, collection_name: str) -> int:
        try:
            client = self._get_client()
            res = client.count(collection_name)
            client.close()
            return res.count
        except Exception:
            return 0

    def search_vector(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 20,
        doc_name: str | None = None,
    ) -> list[RetrievedChunk]:
        query_filter = None
        if doc_name:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=doc_name),
                    )
                ]
            )

        client = self._get_client()
        try:
            hits = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            ).points
        finally:
            client.close()

        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            p = hit.payload or {}
            results.append(
                RetrievedChunk(
                    id=p.get("chunk_id", str(hit.id)),
                    doc_name=p.get("doc_name", ""),
                    page_number=p.get("page_number", 0),
                    page_num_1based=p.get("page_num_1based", 1),
                    chunk_index=p.get("chunk_index", 0),
                    chunk_mode=p.get("chunk_mode", ""),
                    text=p.get("text", ""),
                    raw_content=p.get("raw_content", ""),
                    score=float(hit.score) if hit.score is not None else 0.0,
                    vector_score=float(hit.score) if hit.score is not None else 0.0,
                    rank=rank,
                    metadata=p.get("metadata", {}),
                )
            )

        return results


class BM25IndexManager:
    """
    Manages local BM25 lexical index over chunks.
    """

    def __init__(self, index_file: Path | None = None):
        self.index_file = index_file or (settings.BM25_INDEX_DIR / "production_bm25.pkl")
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self._load_if_exists()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _load_if_exists(self) -> None:
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data["chunks"]
                    self.bm25 = data["bm25"]
                logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")

    def build_and_save(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [self.tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
        logger.info(f"Saved BM25 index with {len(chunks)} chunks to {self.index_file}")

    def search(
        self,
        query: str,
        limit: int = 20,
        doc_name: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair scores with chunks
        indexed_scores = []
        for i, score in enumerate(scores):
            chunk = self.chunks[i]
            if doc_name and chunk.doc_name != doc_name:
                continue
            indexed_scores.append((score, chunk))

        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        top = indexed_scores[:limit]

        results: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(top, start=1):
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    doc_name=chunk.doc_name,
                    page_number=chunk.page_number,
                    page_num_1based=chunk.page_num_1based,
                    chunk_index=chunk.chunk_index,
                    chunk_mode=chunk.chunk_mode,
                    text=chunk.text,
                    raw_content=chunk.raw_content,
                    score=float(score),
                    bm25_score=float(score),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results


class HybridRetriever:
    """
    Unified Hybrid Retriever supporting:
    - Baseline: Dense Vector Search only (Top 5) over naive chunks.
    - Production: Metadata Pre-filter + Hybrid Retrieval (Dense Top-20 + BM25 Top-20)
                  combined via Reciprocal Rank Fusion (RRF), followed by
                  FlashRank Cross-Encoder reranking down to Top 5.
    """

    def __init__(self):
        self.embedding_mgr = EmbeddingManager()
        self.vector_store = QdrantVectorStore()
        self.bm25_mgr = BM25IndexManager()
        self.ranker = Ranker(
            model_name=settings.RERANKER_MODEL_NAME,
            cache_dir=settings.FLASHRANK_CACHE_DIR,
        )

    @staticmethod
    def _calibrate_flashrank_score(raw_score: float) -> float:
        """
        Calibrates FlashRank raw probability score:
        MS-MARCO pairwise cross-encoder raw scores sit around 2e-5 for noise
        and scale up to ~0.05-0.90 for high relevance.
        This maps:
          noise floor (~2e-5) -> ~0.17 (< 0.30 threshold, abstains)
          meaningful signal (~1e-4) -> ~0.38 (> 0.30 threshold, accepted)
          strong signal (>= 0.01) -> ~0.93 - 0.99
        """
        raw_score = max(1e-9, min(1.0 - 1e-9, float(raw_score)))
        logit = math.log(raw_score / (1.0 - raw_score))
        calibrated = 1.0 / (1.0 + math.exp(-(logit + 8.5) / 1.5))
        return float(np.clip(calibrated, 0.0, 1.0))

    def baseline_search(
        self,
        query: str,
        limit: int | None = None,
        doc_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Baseline mode: Dense Vector Search only over naive chunks."""
        k = limit or settings.TOP_K_RERANK
        query_vector = self.embedding_mgr.embed_query(query)
        return self.vector_store.search_vector(
            collection_name=QdrantVectorStore.BASELINE_COLLECTION,
            query_vector=query_vector,
            limit=k,
            doc_name=doc_name,
        )

    def production_search(
        self,
        query: str,
        limit: int | None = None,
        top_k_retrieval: int | None = None,
        doc_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Production mode:
        1. Dense Vector Search Top-20 (over contextual chunks).
        2. BM25 Lexical Search Top-20 (over contextual chunks).
        3. Reciprocal Rank Fusion (RRF).
        4. FlashRank Cross-Encoder reranking down to Top-5.
        """
        k_retrieve = top_k_retrieval or settings.TOP_K_RETRIEVAL
        k_final = limit or settings.TOP_K_RERANK
        rrf_k = settings.RRF_K

        # Step 1: Dense Vector Retrieval Top-20
        query_vector = self.embedding_mgr.embed_query(query)
        dense_hits = self.vector_store.search_vector(
            collection_name=QdrantVectorStore.PRODUCTION_COLLECTION,
            query_vector=query_vector,
            limit=k_retrieve,
            doc_name=doc_name,
        )

        # Step 2: BM25 Lexical Retrieval Top-20
        bm25_hits = self.bm25_mgr.search(
            query=query,
            limit=k_retrieve,
            doc_name=doc_name,
        )

        # Step 3: Reciprocal Rank Fusion (RRF)
        chunk_map: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            chunk_map[hit.id] = hit
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, hit in enumerate(bm25_hits, start=1):
            if hit.id not in chunk_map:
                chunk_map[hit.id] = hit
            else:
                chunk_map[hit.id].bm25_score = hit.bm25_score
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort candidates by RRF score
        sorted_by_rrf = sorted(
            chunk_map.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        top_fused_candidates: list[RetrievedChunk] = []
        for cid in sorted_by_rrf[:k_retrieve]:
            chunk = chunk_map[cid]
            chunk.rrf_score = rrf_scores[cid]
            top_fused_candidates.append(chunk)

        if not top_fused_candidates:
            return []

        # Step 4: FlashRank Cross-Encoder Reranking
        passages = [{"id": c.id, "text": c.text} for c in top_fused_candidates]
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_req)

        scored_candidates: list[RetrievedChunk] = []
        for r in reranked_results:
            cid = r["id"]
            raw_s = float(r["score"])
            calibrated_s = self._calibrate_flashrank_score(raw_s)

            candidate = chunk_map[cid]
            candidate.raw_score = raw_s
            vec_s = candidate.vector_score or 0.0
            # Blend cross-encoder score with dense vector similarity
            candidate.score = float(np.clip(0.5 * calibrated_s + 0.5 * vec_s, 0.0, 1.0))
            scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda c: c.score, reverse=True)
        final_chunks: list[RetrievedChunk] = []
        for rank, candidate in enumerate(scored_candidates[:k_final], start=1):
            candidate.rank = rank
            final_chunks.append(candidate)

        return final_chunks

    def search(
        self,
        query: str,
        mode: Literal["baseline", "production"] = "production",
        doc_name: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        if mode == "baseline":
            return self.baseline_search(query=query, limit=limit, doc_name=doc_name)
        return self.production_search(query=query, limit=limit, doc_name=doc_name)
