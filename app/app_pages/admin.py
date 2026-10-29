import time
import streamlit as st

from app.shared import get_generator, get_retriever
from src.config import settings
from src.evaluation.evaluate import run_ablation_benchmark
from src.ingest import run_full_ingestion
from src.retrieval.hybrid_retriever import QdrantVectorStore

st.title("System Administration Console", icon=":material/admin_panel_settings:")
st.caption(
    "Restricted administrative operations: Index management, vector store health, benchmark execution, and API validation."
)

retriever = get_retriever()
generator = get_generator()

# System Metrics
st.markdown("### 📊 Database & Index Health")

b_count = retriever.vector_store.count(QdrantVectorStore.BASELINE_COLLECTION)
p_count = retriever.vector_store.count(QdrantVectorStore.PRODUCTION_COLLECTION)
bm25_count = len(retriever.bm25_mgr.chunks)
pdf_count = len(list(settings.RAW_DIR.glob("*.pdf"))) if settings.RAW_DIR.exists() else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Raw SEC PDFs", f"{pdf_count} files")
with col2:
    st.metric("Baseline Chunks", f"{b_count:,}")
with col3:
    st.metric("Production Chunks", f"{p_count:,}")
with col4:
    st.metric("BM25 Inverted Chunks", f"{bm25_count:,}")

st.divider()

# Administrative Actions
st.markdown("### 🛠️ Ingestion & Maintenance Tasks")

col_act1, col_act2 = st.columns(2)

with col_act1:
    with st.container(border=True):
        st.markdown("#### Ingestion Pipeline")
        st.caption("Re-parse all 20 SEC PDFs, generate FastEmbed embeddings, and rebuild the BM25 inverted index.")

        if st.button("Re-Ingest All SEC Filings", icon=":material/sync:", type="primary", width="stretch"):
            with st.spinner("Processing SEC PDFs and upserting chunks..."):
                t0 = time.time()
                run_full_ingestion()
                dur = time.time() - t0
                st.success(f"Ingestion completed in {dur:.1f}s! Indices updated.", icon=":material/check_circle:")
                st.rerun()

with col_act2:
    with st.container(border=True):
        st.markdown("#### Evaluation Benchmark")
        st.caption("Execute the full 74-query retrieval ablation benchmark across Baseline and Production pipelines.")

        if st.button("Execute Full Benchmark", icon=":material/play_circle:", width="stretch"):
            with st.spinner("Running 74-query evaluation on FinanceBench..."):
                t0 = time.time()
                run_ablation_benchmark(use_doc_filter=True)
                dur = time.time() - t0
                st.success(f"Benchmark completed in {dur:.1f}s! Results updated.", icon=":material/check_circle:")
                st.rerun()

st.markdown("### 🧹 Cache & Runtime Controls")
col_c1, col_c2 = st.columns(2)
with col_c1:
    if st.button("Clear Streamlit In-Memory Caches", icon=":material/delete_sweep:", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("All caches cleared!", icon=":material/check:")

with col_c2:
    if st.button("Exit Admin Mode", icon=":material/logout:", width="stretch"):
        st.session_state.is_admin = False
        st.query_params.clear()
        st.rerun()
