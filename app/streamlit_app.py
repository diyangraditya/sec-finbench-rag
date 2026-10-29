import sys
from pathlib import Path
import streamlit as st

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# App-level page configuration
st.set_page_config(
    page_title="FinanceBench SEC RAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Admin authentication check via query param or session state
# E.g., navigating to http://localhost:8501/?admin=true
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if st.query_params.get("admin") in ["true", "1"]:
    st.session_state.is_admin = True

# Define Navigation Structure
pages = {
    "": [
        st.Page(
            "app_pages/chat.py",
            title="Financial Analyst",
            icon=":material/chat:",
            default=True,
        ),
    ],
    "Technical Deep Dive": [
        st.Page(
            "app_pages/benchmarks.py",
            title="Evaluation Benchmarks",
            icon=":material/query_stats:",
        ),
        st.Page(
            "app_pages/comparator.py",
            title="Baseline vs. Production",
            icon=":material/compare_arrows:",
        ),
        st.Page(
            "app_pages/architecture.py",
            title="System Architecture",
            icon=":material/schema:",
        ),
    ],
}

# Dynamically add Admin page if admin access is granted
if st.session_state.is_admin:
    pages["System Operations"] = [
        st.Page(
            "app_pages/admin.py",
            title="Admin Console",
            icon=":material/admin_panel_settings:",
        ),
    ]

# Setup multi-page navigation in sidebar
nav = st.navigation(pages, position="sidebar")

# Sidebar metadata and footer controls
with st.sidebar:
    st.caption("### 📈 FinanceBench SEC RAG")
    st.markdown(
        """
        **Corpus:** 20 SEC 10-K Filings  
        **Chunks:** 5,391 Contextual Chunks  
        **Embeddings:** BAAI/bge-small-en-v1.5  
        **Reranker:** ms-marco-MiniLM-L-12-v2  
        **LLM:** Gemini 2.5 Flash / Groq Fallback  
        """
    )

    st.divider()

    # Admin access toggle in sidebar footer
    if not st.session_state.is_admin:
        with st.popover("Operator Access", icon=":material/lock:"):
            st.caption("Access the Admin Console to re-index documents or run evaluations.")
            passcode = st.text_input("Enter Passcode", type="password", key="admin_pass_input")
            if st.button("Unlock Admin View", icon=":material/key:", width="stretch"):
                # Default convenient dev passcode or empty accepts
                if passcode in ["admin", "secret", ""]:
                    st.session_state.is_admin = True
                    st.query_params["admin"] = "true"
                    st.rerun()
                else:
                    st.error("Invalid passcode")
            st.caption("Tip: You can also append `?admin=true` to your URL.")
    else:
        st.success("Admin Mode Active", icon=":material/verified_user:")
        if st.button("Exit Admin Mode", icon=":material/logout:", width="stretch"):
            st.session_state.is_admin = False
            st.query_params.clear()
            st.rerun()

# Execute the selected page
nav.run()
