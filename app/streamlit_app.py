import json
import logging
import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.generation.generator import RAGGenerator, RAGResponse
from src.ingestion.parser_and_chunker import (
    ingest_all_documents,
    parse_pdf_to_pages,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    QdrantVectorStore,
    RetrievedChunk,
)
from src.evaluation.evaluate import run_ablation_benchmark

# Page setup
st.set_page_config(
    page_title="FinanceBench RAG Maxxing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .badge-provider {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-gemini { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
    .badge-groq { background: #701a75; color: #f0abfc; border: 1px solid #d946ef; }
    .badge-guard { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .badge-abstain { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }

    .card-container {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1e293b;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-container:hover {
        border-color: #38bdf8;
    }

    .metric-pill {
        background: #1e293b;
        border-radius: 8px;
        padding: 8px 12px;
        margin-right: 8px;
        display: inline-block;
        font-size: 0.85rem;
        border: 1px solid #334155;
    }

    .chunk-box {
        background-color: #111827;
        border-left: 4px solid #6366f1;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }

    .ground-truth-box {
        background: linear-gradient(135deg, #14291f 0%, #0f172a 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_retriever():
    return HybridRetriever()


@st.cache_resource
def get_generator():
    return RAGGenerator()


@st.cache_data
def load_eval_dataset():
    path = settings.EVAL_DATASET_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


retriever = get_retriever()
generator = get_generator()
eval_dataset = load_eval_dataset()

# Top Header Banner
st.markdown(
    """
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; color: #f8fafc; font-size: 1.8rem; font-weight: 700;">
                📈 FinanceBench Production RAG & Benchmark
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                SEC 10-K Retrieval-Augmented Generation | Baseline (Dense) vs. Production (Hybrid RRF + FlashRank)
            </p>
        </div>
        <div>
            <span class="badge-provider badge-gemini">Gemini Primary</span>
            <span class="badge-provider badge-groq">Groq Fallback</span>
            <span class="badge-provider badge-guard">Qdrant + BM25</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.subheader("⚙️ System Control & Stats")

    b_count = retriever.vector_store.count(QdrantVectorStore.BASELINE_COLLECTION)
    p_count = retriever.vector_store.count(QdrantVectorStore.PRODUCTION_COLLECTION)
    bm25_count = len(retriever.bm25_mgr.chunks)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Baseline Chunks", b_count)
    with col_s2:
        st.metric("Production Chunks", p_count)

    st.metric("BM25 Indexed Chunks", bm25_count)

    st.divider()

    st.markdown("### 🤖 LLM Configuration")
    provider_choice = st.selectbox(
        "Active LLM Provider",
        options=["gemini", "groq"],
        index=0 if settings.LLM_PROVIDER == "gemini" else 1,
        help="Primary provider for generation. Automatic fallback engages on 429 rate limits.",
    )

    doc_filter_opt = st.selectbox(
        "Metadata Pre-filter (Doc)",
        options=["(All Documents)"]
        + sorted(list({item["doc_name"] for item in eval_dataset})),
        index=0,
    )
    doc_filter = None if doc_filter_opt == "(All Documents)" else doc_filter_opt

    top_k_val = st.slider("Passages to Rerank (Top-K)", min_value=1, max_value=10, value=5)

    st.divider()

    st.markdown("### 🛠️ Admin Actions")
    if st.button("🚀 Ingest All SEC Filings", use_container_width=True):
        with st.spinner("Ingesting & embedding PDFs into Qdrant & BM25..."):
            t0 = time.time()
            naive_chunks, contextual_chunks = ingest_all_documents(settings.RAW_DIR)

            # Ingest Baseline
            naive_texts = [c.text for c in naive_chunks]
            naive_embs = retriever.embedding_mgr.embed_texts(naive_texts)
            retriever.vector_store.upsert_chunks(
                collection_name=QdrantVectorStore.BASELINE_COLLECTION,
                chunks=naive_chunks,
                embeddings=naive_embs,
            )

            # Ingest Production
            contextual_texts = [c.text for c in contextual_chunks]
            contextual_embs = retriever.embedding_mgr.embed_texts(contextual_texts)
            retriever.vector_store.upsert_chunks(
                collection_name=QdrantVectorStore.PRODUCTION_COLLECTION,
                chunks=contextual_chunks,
                embeddings=contextual_embs,
            )

            # Build BM25
            retriever.bm25_mgr.build_and_save(contextual_chunks)
            dur = time.time() - t0
            st.success(f"Ingestion complete in {dur:.1f}s! Reloading page...")
            st.rerun()

    if st.button("📊 Run Retrieval Ablation Benchmark", use_container_width=True):
        with st.spinner("Running 30-query ablation benchmark..."):
            run_ablation_benchmark()
            st.success("Benchmark completed! See 'Evaluation Benchmark' tab.")
            st.rerun()

# Main Tabs
tab_compare, tab_benchmark = st.tabs(["⚡ Side-by-Side RAG Comparator", "📊 Evaluation Benchmark Report"])

with tab_compare:
    st.subheader("Select a Benchmark Question or Ask Your Own")

    col_q1, col_q2 = st.columns([1, 2])
    with col_q1:
        doc_options = ["(All Companies)"] + sorted(list({item["company"] for item in eval_dataset}))
        selected_company = st.selectbox("Filter queries by Company", options=doc_options)

    filtered_samples = [
        item
        for item in eval_dataset
        if selected_company == "(All Companies)" or item["company"] == selected_company
    ]

    query_labels = [
        f"[{item['company']} | {item['doc_name']}] {item['question']}"
        for item in filtered_samples
    ]

    CUSTOM_OPTION = "✏️ (Custom Question — Type your own below)"
    options_list = [CUSTOM_OPTION] + query_labels

    with col_q2:
        selected_query_label = st.selectbox(
            "Choose a Preset Question (or pick Custom):",
            options=options_list,
            index=0,
            help="Select a preset benchmark question to inspect ground truth, or pick 'Custom Question' to ask anything.",
        )

    selected_sample = None
    default_text = ""
    if selected_query_label != CUSTOM_OPTION and filtered_samples:
        selected_idx = query_labels.index(selected_query_label)
        selected_sample = filtered_samples[selected_idx]
        default_text = selected_sample["question"]

    user_query = st.text_area(
        "Enter Financial Question:",
        value=default_text,
        placeholder="Type any custom question (e.g. What were Boeing's total liabilities in 2022? Or what are AMD's primary risk factors?)...",
        height=85,
        help="You can ask ANY arbitrary question about the SEC 10-K filings in the database.",
    )

    run_btn = st.button("🔍 Run Side-by-Side Comparison", type="primary", use_container_width=True)

    if run_btn and user_query.strip():
        with st.spinner("Executing Baseline and Production RAG pipelines..."):
            query_str = user_query.strip()
            active_doc_filter = (
                doc_filter
                or (selected_sample["doc_name"] if selected_sample else None)
            )

            # 1. Baseline RAG Run
            t0_base = time.time()
            t_base_ret = time.time()
            base_chunks = retriever.search(
                query=query_str,
                mode="baseline",
                doc_name=active_doc_filter,
                limit=top_k_val,
            )
            lat_base_ret = time.time() - t_base_ret
            base_resp, lat_base_gen = generator.generate(
                question=query_str,
                chunks=base_chunks,
                provider=provider_choice,
                enforce_abstention=False,
            )
            lat_base_total = time.time() - t0_base

            # 2. Production RAG Run
            t0_prod = time.time()
            t_prod_ret = time.time()
            prod_chunks = retriever.search(
                query=query_str,
                mode="production",
                doc_name=active_doc_filter,
                limit=top_k_val,
            )
            lat_prod_ret = time.time() - t_prod_ret
            prod_resp, lat_prod_gen = generator.generate(
                question=query_str,
                chunks=prod_chunks,
                provider=provider_choice,
                enforce_abstention=True,
            )
            lat_prod_total = time.time() - t0_prod

        # Render Side-by-Side Results
        col_base, col_prod = st.columns(2)

        # ----------------- BASELINE COLUMN -----------------
        with col_base:
            st.markdown("### 🔹 Baseline RAG")
            st.caption("Dense Vector Search Only (Naive Chunks, No Reranker, No Context Headers)")

            st.markdown(
                f"""
            <div class="card-container">
                <div style="margin-bottom: 10px;">
                    <span class="metric-pill">⏱️ Total: <b>{lat_base_total:.2f}s</b></span>
                    <span class="metric-pill">Ret: {lat_base_ret*1000:.0f}ms</span>
                    <span class="metric-pill">Gen: {lat_base_gen:.2f}s</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="badge-provider badge-gemini">{base_resp.provider_used}</span>
                    <span class="badge-provider {'badge-guard' if base_resp.is_grounded else 'badge-abstain'}">
                        {'Grounded' if base_resp.is_grounded else 'Ungrounded / Hallucination Risk'}
                    </span>
                </div>
                <div style="font-size: 1.02rem; line-height: 1.6; color: #f1f5f9;">
                    {base_resp.answer}
                </div>
                <div style="margin-top: 14px; font-size: 0.85rem; color: #94a3b8;">
                    <b>Cited Pages:</b> {base_resp.cited_pages or 'None'} | <b>Docs:</b> {base_resp.cited_doc_names or 'None'}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Inspect Top-{len(base_chunks)} Retrieved Chunks (Baseline)"):
                for rank, c in enumerate(base_chunks, start=1):
                    st.markdown(
                        f"""
                    <div class="chunk-box" style="border-left-color: #3b82f6;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <b>Rank #{rank} | Doc: {c.doc_name} | Page: {c.page_number} (1-based: {c.page_num_1based})</b>
                            <span>Vector Sim: <b>{c.vector_score:.3f}</b></span>
                        </div>
                        <div style="color: #cbd5e1; white-space: pre-wrap; font-family: monospace; font-size: 0.82rem;">{c.text[:400]}...</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # ----------------- PRODUCTION COLUMN -----------------
        with col_prod:
            st.markdown("### 🚀 Production Hybrid RAG")
            st.caption("Hybrid RRF (Dense + BM25) + FlashRank Cross-Encoder + Context Headers + Abstention Guard")

            abstain_badge = (
                '<span class="badge-provider badge-abstain">Abstention Triggered (< 0.30)</span>'
                if not prod_resp.is_grounded
                else '<span class="badge-provider badge-guard">Verified & Grounded</span>'
            )

            st.markdown(
                f"""
            <div class="card-container" style="border-color: #6366f1;">
                <div style="margin-bottom: 10px;">
                    <span class="metric-pill">⏱️ Total: <b>{lat_prod_total:.2f}s</b></span>
                    <span class="metric-pill">Ret: {lat_prod_ret*1000:.0f}ms</span>
                    <span class="metric-pill">Gen: {lat_prod_gen:.2f}s</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="badge-provider badge-groq">{prod_resp.provider_used}</span>
                    {abstain_badge}
                </div>
                <div style="font-size: 1.02rem; line-height: 1.6; color: #f1f5f9;">
                    {prod_resp.answer}
                </div>
                <div style="margin-top: 14px; font-size: 0.85rem; color: #94a3b8;">
                    <b>Cited Pages:</b> {prod_resp.cited_pages or 'None'} | <b>Docs:</b> {prod_resp.cited_doc_names or 'None'}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Inspect Top-{len(prod_chunks)} Retrieved Chunks (Production)"):
                for rank, c in enumerate(prod_chunks, start=1):
                    raw_s_str = f"{c.raw_score:.6f}" if c.raw_score is not None else "N/A"
                    rrf_s_str = f"{c.rrf_score:.4f}" if c.rrf_score is not None else "N/A"
                    st.markdown(
                        f"""
                    <div class="chunk-box" style="border-left-color: #10b981;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <b>Rank #{rank} | Doc: {c.doc_name} | Page: {c.page_number} (1-based: {c.page_num_1based})</b>
                            <span>Rerank Score: <b style="color: #34d399;">{c.score:.3f}</b> (Raw: {raw_s_str})</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;">
                            RRF Score: {rrf_s_str} | BM25: {c.bm25_score if c.bm25_score else 'N/A'}
                        </div>
                        <div style="color: #cbd5e1; white-space: pre-wrap; font-family: monospace; font-size: 0.82rem;">{c.text[:450]}...</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # ----------------- GROUND TRUTH COMPARISON -----------------
        is_preset_eval = (
            selected_sample is not None
            and user_query.strip().lower() == selected_sample.get("question", "").strip().lower()
        )
        if is_preset_eval:
            st.markdown("---")
            gt_pages = selected_sample.get("evidence_pages") or []
            base_hit = any(
                p in gt_pages and c.doc_name == selected_sample["doc_name"]
                for p, c in zip([c.page_number for c in base_chunks], base_chunks)
            )
            prod_hit = any(
                p in gt_pages and c.doc_name == selected_sample["doc_name"]
                for p, c in zip([c.page_number for c in prod_chunks], prod_chunks)
            )

            col_gt1, col_gt2 = st.columns([1, 1])
            with col_gt1:
                st.markdown(
                    f"""
                <div class="ground-truth-box">
                    <h4 style="margin: 0 0 10px 0; color: #34d399;">🎯 FinanceBench Ground Truth</h4>
                    <p style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.5;">
                        <b>Official Answer:</b> {selected_sample.get('ground_truth_answer')}
                    </p>
                    <div style="font-size: 0.85rem; color: #94a3b8;">
                        <b>Expected Evidence Pages:</b> {gt_pages} | <b>Doc:</b> {selected_sample.get('doc_name')}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col_gt2:
                st.markdown(
                    f"""
                <div class="card-container" style="margin-top: 20px;">
                    <h4 style="margin: 0 0 10px 0; color: #38bdf8;">🏁 Retrieval Accuracy Check</h4>
                    <div style="margin-bottom: 8px;">
                        Baseline Page Hit: <b style="color: {'#34d399' if base_hit else '#f87171'};">
                            {'✅ HIT' if base_hit else '❌ MISS'}
                        </b>
                    </div>
                    <div>
                        Production Page Hit: <b style="color: {'#34d399' if prod_hit else '#f87171'};">
                            {'✅ HIT' if prod_hit else '❌ MISS'}
                        </b>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# Tab 2: Evaluation Benchmark
with tab_benchmark:
    st.subheader("Automated Retrieval Ablation Benchmark Results")

    ablation_md_path = settings.DATA_DIR / "ablation_results.md"
    ablation_json_path = settings.DATA_DIR / "ablation_results.json"

    if ablation_json_path.exists():
        with open(ablation_json_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        b_sum = bench_data.get("baseline_summary", {})
        p_sum = bench_data.get("production_summary", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Exact Page Hit Rate @ 5",
                f"{p_sum.get('page_hit_rate_at_5', 0)*100:.1f}%",
                delta=f"{(p_sum.get('page_hit_rate_at_5', 0) - b_sum.get('page_hit_rate_at_5', 0))*100:+.1f}% vs Baseline",
            )
        with m2:
            st.metric(
                "Mean Reciprocal Rank (MRR@5)",
                f"{p_sum.get('mrr_at_5', 0):.3f}",
                delta=f"{p_sum.get('mrr_at_5', 0) - b_sum.get('mrr_at_5', 0):+.3f} vs Baseline",
            )
        with m3:
            st.metric(
                "Document Hit Rate @ 5",
                f"{p_sum.get('doc_hit_rate_at_5', 0)*100:.1f}%",
                delta=f"{(p_sum.get('doc_hit_rate_at_5', 0) - b_sum.get('doc_hit_rate_at_5', 0))*100:+.1f}% vs Baseline",
            )
        with m4:
            st.metric(
                "Average Retrieval Latency",
                f"{p_sum.get('avg_latency_sec', 0)*1000:.0f} ms",
                delta=f"{(p_sum.get('avg_latency_sec', 0) - b_sum.get('avg_latency_sec', 0))*1000:+.0f} ms",
                delta_color="inverse",
            )

    if ablation_md_path.exists():
        with open(ablation_md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        st.markdown(md_content)
    else:
        st.info("No benchmark results found yet. Run ingestion and click 'Run Retrieval Ablation Benchmark' to generate results!")
