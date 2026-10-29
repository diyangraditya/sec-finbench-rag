from pathlib import Path
from typing import Literal
# pyrefly: ignore[missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Provider & Model Settings
    LLM_PROVIDER: Literal["gemini", "groq"] = "gemini"
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_FALLBACK_MODELS: list[str] = [
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-flash-latest",
    ]
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FALLBACK_MODELS: list[str] = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
    ]
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_DIM: int = 384  

    # Retrieval & Reranker Settings
    TOP_K_RETRIEVAL: int = 25
    TOP_K_RERANK: int = 5
    RERANK_SCORE_THRESHOLD: float = 0.30
    RRF_K: int = 60
    RERANKER_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"

    # Directory Paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    EVAL_DATASET_PATH: Path = BASE_DIR / "data" / "eval_dataset.json"
    QDRANT_STORAGE_PATH: str = str(BASE_DIR / "qdrant_storage")
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"
    EMBEDDING_CACHE_DB: Path = BASE_DIR / "data" / "cache" / "embedding_cache.sqlite"
    BM25_INDEX_DIR: Path = BASE_DIR / "data" / "bm25_indices"
    FLASHRANK_CACHE_DIR: str = str(BASE_DIR / "flashrank_cache")

    def ensure_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.QDRANT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.FLASHRANK_CACHE_DIR).mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
