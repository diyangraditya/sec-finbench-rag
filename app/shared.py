import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from src.config import settings
from src.generation.generator import RAGGenerator, RAGResponse
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    QdrantVectorStore,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Initializing Hybrid Retriever (FastEmbed + BM25 + FlashRank)...")
def get_retriever() -> HybridRetriever:
    """Shared singleton instance of the HybridRetriever."""
    return HybridRetriever()


@st.cache_resource(show_spinner="Initializing RAG Generator (Gemini + Groq Fallback)...")
def get_generator() -> RAGGenerator:
    """Shared singleton instance of the RAGGenerator."""
    return RAGGenerator()


@st.cache_data(show_spinner=False)
def load_eval_dataset() -> List[Dict[str, Any]]:
    """Loads the FinanceBench evaluation dataset."""
    path = settings.EVAL_DATASET_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}")
    return []


@st.cache_data(show_spinner=False)
def load_ablation_results() -> Dict[str, Any]:
    """Loads the ablation results JSON if available."""
    json_path = settings.DATA_DIR / "ablation_results.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ablation results: {e}")
    return {}


@st.cache_data(show_spinner=False)
def load_ablation_markdown() -> str:
    """Loads the ablation report markdown if available."""
    md_path = settings.DATA_DIR / "ablation_results.md"
    if md_path.exists():
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load ablation report: {e}")
    return ""


def get_available_companies() -> List[str]:
    """Extracts unique company names from the evaluation dataset."""
    dataset = load_eval_dataset()
    companies = sorted(list({item.get("company", "") for item in dataset if item.get("company")}))
    return companies


def get_available_documents() -> List[str]:
    """Extracts unique document filenames from the evaluation dataset."""
    dataset = load_eval_dataset()
    docs = sorted(list({item.get("doc_name", "") for item in dataset if item.get("doc_name")}))
    return docs
