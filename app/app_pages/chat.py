import time
import streamlit as st

from app.shared import (
    get_available_companies,
    get_generator,
    get_retriever,
    load_eval_dataset,
)

# Page header
st.title("SEC Financial Analyst", icon=":material/account_balance:")
st.caption(
    "Grounded conversational analysis over 20 SEC 10-K filings (AMD, Boeing, Microsoft, Amazon, Best Buy, etc.) "
    "powered by Hybrid RRF Retrieval, FlashRank Cross-Encoder, and Abstention Guardrails."
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Top controls row: Company filter & Clear chat
companies = get_available_companies()
col_filter, col_clear = st.columns([3, 1], vertical_alignment="bottom")

with col_filter:
    selected_company = st.selectbox(
        "Focus on Company (Optional)",
        options=["(All 20 Companies)"] + companies,
        index=0,
        help="Filter retrieval exclusively to a specific company's 10-K filing, or search across all 20 companies.",
    )

with col_clear:
    if st.button("Clear History", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# Determine document filter (supports multiple filings per company, e.g. 3M, Best Buy, Pfizer)
eval_data = load_eval_dataset()
company_docs_map = {}
for item in eval_data:
    comp = item.get("company")
    doc = item.get("doc_name")
    if comp and doc:
        company_docs_map.setdefault(comp, set()).add(doc)

doc_filter = None
if selected_company != "(All 20 Companies)" and selected_company in company_docs_map:
    doc_filter = list(company_docs_map[selected_company])

# Curated suggestions for FinanceBench filings
SUGGESTIONS = {
    ":blue[:material/memory:] AMD Major Products": "What are the major products that AMD sells?",
    ":green[:material/flight:] Boeing 2022 Liabilities": "What were Boeing's total liabilities in 2022?",
    ":orange[:material/storefront:] Best Buy FY2023 CapEx": "What was Best Buy's FY 2023 capital expenditures?",
    ":violet[:material/trending_up:] Adobe Digital Media ARR": "What was Adobe's Digital Media ARR at the end of FY2022?",
    ":red[:material/shopping_cart:] Amazon 2022 Net Sales": "What was Amazon's total net sales in 2022?",
}

# Display existing chat history
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = ":material/person:" if role == "user" else ":material/account_balance:"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and msg.get("evidence"):
            num_passages = len(msg["evidence"])
            unique_docs = len({c.get("doc_name") for c in msg["evidence"] if c.get("doc_name")})
            with st.expander(
                f":material/find_in_page: Retrieved {num_passages} SEC Passages across {unique_docs} Filings",
                expanded=False,
            ):
                for rank, c in enumerate(msg["evidence"], start=1):
                    col_meta, col_score = st.columns([3, 1])
                    with col_meta:
                        table_tag = " · *[Contains Table]*" if c.get("has_table") else ""
                        p_0 = c.get("page_number", 0)
                        p_1 = c.get("page_num_1based", p_0 + 1)
                        st.markdown(
                            f"**#{rank} {c.get('doc_name')}** · *Page {p_0} (PDF p.{p_1})*"
                            f"{table_tag}"
                        )
                    with col_score:
                        st.caption(f"Score: `{c.get('score', 0):.3f}`")
                    st.code(c.get("text", "")[:350] + ("..." if len(c.get("text", "")) > 350 else ""), language="text")

        # Visual alert if this response was an abstention
        if role == "assistant" and not msg.get("is_grounded", True):
            st.warning(
                "🛡️ **Abstention Guardrail Triggered:** Retrieval confidence was below the safety threshold (>= 0.30). "
                "The response abstained to prevent financial hallucination.",
                icon=":material/shield:",
            )

        st.markdown(msg["content"])

        if role == "assistant":
            meta_items = []
            if msg.get("cited_pages"):
                pages_str = ", ".join(str(p) for p in msg["cited_pages"])
                meta_items.append(f":material/bookmark: **Pages:** {pages_str}")
            if msg.get("provider"):
                meta_items.append(f":material/bolt: **Provider:** {msg['provider']}")
            if msg.get("latency"):
                meta_items.append(f":material/timer: **Latency:** {msg['latency']:.2f}s")
            if meta_items:
                st.caption(" · ".join(meta_items))

# Suggestion Chips (only displayed when conversation is empty)
prompt = st.chat_input("Ask a financial question about SEC 10-K filings...", submit_mode="disable")

if not st.session_state.messages:
    suggestions_placeholder = st.empty()
    selected_suggestion = suggestions_placeholder.pills(
        "Try asking:",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
    )
    if selected_suggestion:
        prompt = SUGGESTIONS[selected_suggestion]
    if prompt:
        suggestions_placeholder.empty()

# Process User Input
if prompt:
    # 1. Append and render User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    # 2. Retrieve & Generate Assistant response
    with st.chat_message("assistant", avatar=":material/account_balance:"):
        retriever = get_retriever()
        generator = get_generator()

        with st.status(":shimmer[Retrieving SEC 10-K Filings & Reranking...]", type="compact") as status:
            t0 = time.time()
            t_ret0 = time.time()
            chunks = retriever.search(
                query=prompt,
                mode="production",
                doc_name=doc_filter,
                limit=5,
            )
            lat_ret = time.time() - t_ret0

            num_docs = len({c.doc_name for c in chunks}) if chunks else 0
            status.update(
                label=f"Retrieved {len(chunks)} passages across {num_docs} filings ({lat_ret*1000:.0f}ms)",
                state="complete",
            )

        # Show evidence expander safely without assuming specific attributes
        evidence_data = []
        for c in chunks:
            c_meta = getattr(c, "metadata", {}) or {}
            evidence_data.append(
                {
                    "doc_name": getattr(c, "doc_name", ""),
                    "page_number": getattr(c, "page_number", 0),
                    "page_num_1based": getattr(c, "page_num_1based", getattr(c, "page_number", 0) + 1),
                    "score": getattr(c, "score", 0.0),
                    "section": getattr(c, "section", c_meta.get("section", "General")),
                    "has_table": c_meta.get("has_table", False),
                    "text": getattr(c, "text", ""),
                }
            )

        if evidence_data:
            with st.expander(
                f":material/find_in_page: Retrieved {len(evidence_data)} SEC Passages across {num_docs} Filings",
                expanded=False,
            ):
                for rank, c in enumerate(evidence_data, start=1):
                    col_meta, col_score = st.columns([3, 1])
                    with col_meta:
                        table_tag = " · *[Contains Table]*" if c.get("has_table") else ""
                        p_0 = c.get("page_number", 0)
                        p_1 = c.get("page_num_1based", p_0 + 1)
                        st.markdown(
                            f"**#{rank} {c.get('doc_name')}** · *Page {p_0} (PDF p.{p_1})*"
                            f"{table_tag}"
                        )
                    with col_score:
                        st.caption(f"Score: `{c.get('score', 0):.3f}`")
                    st.code(c.get("text", "")[:350] + ("..." if len(c.get("text", "")) > 350 else ""), language="text")

        # Run generator
        with st.spinner("Synthesizing grounded financial answer..."):
            rag_resp, lat_gen = generator.generate(
                question=prompt,
                chunks=chunks,
                enforce_abstention=True,
            )
            total_lat = time.time() - t0

        # Visual alert if abstention triggered
        if not rag_resp.is_grounded or rag_resp.provider_used == "abstention_guardrail":
            st.warning(
                "🛡️ **Abstention Guardrail Triggered:** Retrieval confidence was below the safety threshold (>= 0.30). "
                "The response abstained to prevent financial hallucination.",
                icon=":material/shield:",
            )

        # Render response content
        st.markdown(rag_resp.answer)

        # Meta captions
        meta_items = []
        if rag_resp.cited_pages:
            pages_str = ", ".join(str(p) for p in rag_resp.cited_pages)
            meta_items.append(f":material/bookmark: **Pages:** {pages_str}")
        if rag_resp.provider_used:
            meta_items.append(f":material/bolt: **Provider:** {rag_resp.provider_used}")
        meta_items.append(f":material/timer: **Total Latency:** {total_lat:.2f}s")
        st.caption(" · ".join(meta_items))

        # Save assistant message to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": rag_resp.answer,
                "evidence": evidence_data,
                "cited_pages": rag_resp.cited_pages,
                "provider": rag_resp.provider_used,
                "is_grounded": rag_resp.is_grounded,
                "latency": total_lat,
            }
        )
