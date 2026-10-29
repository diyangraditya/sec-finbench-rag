import time
import streamlit as st

from app.shared import (
    get_available_companies,
    get_generator,
    get_retriever,
    load_eval_dataset,
)

st.title("Baseline vs. Production Comparator", icon=":material/compare_arrows:")
st.caption(
    "Live side-by-side A/B execution: Compare Naive Dense Vector RAG against Hybrid RRF + FlashRank Cross-Encoder "
    "with FinanceBench Ground Truth verification."
)

eval_dataset = load_eval_dataset()
retriever = get_retriever()
generator = get_generator()

col_comp, col_preset = st.columns([1, 2])
with col_comp:
    companies = ["(All Companies)"] + get_available_companies()
    selected_company = st.selectbox("Filter preset by Company", options=companies)

filtered_samples = [
    item
    for item in eval_dataset
    if selected_company == "(All Companies)" or item.get("company") == selected_company
]

query_labels = [
    f"[{item.get('company', '')} | {item.get('doc_name', '')}] {item.get('question', '')}"
    for item in filtered_samples
]

CUSTOM_OPTION = "✏️ (Custom Question — Type your own below)"
options_list = [CUSTOM_OPTION] + query_labels

with col_preset:
    selected_query_label = st.selectbox(
        "Choose a Preset Benchmark Question",
        options=options_list,
        index=0,
        help="Select a benchmark question to evaluate against ground truth, or pick custom to ask anything.",
    )

selected_sample = None
default_text = ""
if selected_query_label != CUSTOM_OPTION and filtered_samples:
    selected_idx = query_labels.index(selected_query_label)
    selected_sample = filtered_samples[selected_idx]
    default_text = selected_sample.get("question", "")

user_query = st.text_area(
    "Financial Query to Compare:",
    value=default_text,
    placeholder="Enter any financial query (e.g. What were Boeing's total liabilities in 2022?)...",
    height=80,
)

col_run, col_k = st.columns([3, 1], vertical_alignment="bottom")
with col_k:
    top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=5)
with col_run:
    run_btn = st.button("Run Side-by-Side Comparison", type="primary", icon=":material/bolt:", width="stretch")

if run_btn and user_query.strip():
    query_str = user_query.strip()
    active_doc_filter = selected_sample.get("doc_name") if selected_sample else None

    with st.spinner("Executing Baseline and Production pipelines..."):
        # 1. Baseline Run
        t0_base = time.time()
        t_base_ret = time.time()
        base_chunks = retriever.search(
            query=query_str,
            mode="baseline",
            doc_name=active_doc_filter,
            limit=top_k,
        )
        lat_base_ret = time.time() - t_base_ret
        base_resp, lat_base_gen = generator.generate(
            question=query_str,
            chunks=base_chunks,
            enforce_abstention=False,
        )
        lat_base_total = time.time() - t0_base

        # 2. Production Run
        t0_prod = time.time()
        t_prod_ret = time.time()
        prod_chunks = retriever.search(
            query=query_str,
            mode="production",
            doc_name=active_doc_filter,
            limit=top_k,
        )
        lat_prod_ret = time.time() - t_prod_ret
        prod_resp, lat_prod_gen = generator.generate(
            question=query_str,
            chunks=prod_chunks,
            enforce_abstention=True,
        )
        lat_prod_total = time.time() - t0_prod

    # Display side-by-side columns
    col_base, col_prod = st.columns(2)

    # BASELINE COLUMN
    with col_base:
        with st.container(border=True):
            st.markdown("### 🔹 Baseline RAG")
            st.caption("Dense Vector Search Only (Naive Chunks, No Reranker, No Context Headers)")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Latency", f"{lat_base_total:.2f}s")
            with m2:
                st.metric("Retrieval", f"{lat_base_ret*1000:.0f}ms")
            with m3:
                st.metric("Generation", f"{lat_base_gen:.2f}s")

            st.caption(
                f":material/bolt: Provider: **{base_resp.provider_used}** · "
                f":material/shield: Status: **{'Grounded' if base_resp.is_grounded else 'Ungrounded / Risk'}**"
            )

            st.markdown(base_resp.answer)

            pages_str = ", ".join(str(p) for p in base_resp.cited_pages) if base_resp.cited_pages else "None"
            st.caption(f":material/bookmark: **Cited Pages:** {pages_str}")

            with st.expander(f"Inspect Top-{len(base_chunks)} Retrieved Chunks (Baseline)"):
                for rank, c in enumerate(base_chunks, start=1):
                    v_str = f"{c.vector_score:.3f}" if c.vector_score is not None else "N/A"
                    st.markdown(
                        f"**#{rank} | Doc: {c.doc_name} | Page: {c.page_number}** (Vector Sim: `{v_str}`)"
                    )
                    st.code(c.text[:300] + "...", language="text")

    # PRODUCTION COLUMN
    with col_prod:
        with st.container(border=True):
            st.markdown("### 🚀 Production Hybrid RAG")
            st.caption("Hybrid RRF (Dense + BM25) + FlashRank MiniLM-12 Reranker + Abstention Guard")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Latency", f"{lat_prod_total:.2f}s")
            with m2:
                st.metric("Retrieval", f"{lat_prod_ret*1000:.0f}ms")
            with m3:
                st.metric("Generation", f"{lat_prod_gen:.2f}s")

            status_str = "Verified & Grounded" if prod_resp.is_grounded else "Abstention Triggered (< 0.30)"
            st.caption(
                f":material/bolt: Provider: **{prod_resp.provider_used}** · "
                f":material/shield: Status: **{status_str}**"
            )

            st.markdown(prod_resp.answer)

            pages_str = ", ".join(str(p) for p in prod_resp.cited_pages) if prod_resp.cited_pages else "None"
            st.caption(f":material/bookmark: **Cited Pages:** {pages_str}")

            with st.expander(f"Inspect Top-{len(prod_chunks)} Retrieved Chunks (Production)"):
                for rank, c in enumerate(prod_chunks, start=1):
                    raw_str = f"{c.raw_score:.4f}" if c.raw_score is not None else "N/A"
                    st.markdown(
                        f"**#{rank} | Doc: {c.doc_name} | Page: {c.page_number}** "
                        f"(Rerank Score: `{c.score:.3f}` | Raw Logit: `{raw_str}`)"
                    )
                    st.code(c.text[:300] + "...", language="text")

    # GROUND TRUTH VALIDATION (If preset question)
    is_preset = (
        selected_sample is not None
        and user_query.strip().lower() == selected_sample.get("question", "").strip().lower()
    )
    if is_preset:
        st.divider()
        gt_pages = selected_sample.get("evidence_pages") or []
        doc_expected = selected_sample.get("doc_name")

        base_hit = any(
            c.page_number in gt_pages and c.doc_name == doc_expected
            for c in base_chunks
        )
        prod_hit = any(
            c.page_number in gt_pages and c.doc_name == doc_expected
            for c in prod_chunks
        )

        col_gt1, col_gt2 = st.columns([2, 1])
        with col_gt1:
            with st.container(border=True):
                st.markdown("### 🎯 FinanceBench Ground Truth")
                st.markdown(f"**Official Answer:** {selected_sample.get('ground_truth_answer')}")
                st.caption(f"**Expected Evidence Pages:** {gt_pages} | **Document:** {doc_expected}")

        with col_gt2:
            with st.container(border=True):
                st.markdown("### 🏁 Page Hit Verification")
                st.markdown(
                    f"Baseline Hit: **{'✅ HIT' if base_hit else '❌ MISS'}**<br>"
                    f"Production Hit: **{'✅ HIT' if prod_hit else '❌ MISS'}**",
                    unsafe_allow_html=True,
                )
