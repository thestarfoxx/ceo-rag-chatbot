# app.py
import os
import textwrap
from typing import Dict, Any, List

import streamlit as st

# Import your console-based system
from conversation import (
    create_console_rag_system,
    LMSTUDIO_BASE_URL,
)


# -----------------------------
# Session helpers
# -----------------------------
def init_session() -> None:
    """Initialize Streamlit session state."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = create_console_rag_system()
    if "messages" not in st.session_state:
        # Initial assistant message
        welcome = textwrap.dedent(
            f"""
            👋 **Welcome to the CEO RAG Chatbot (Web UI)**

            - Backend: LM Studio at `{LMSTUDIO_BASE_URL}`
            - Abilities:
              - 🔍 Document search (vector-based)
              - 📊 Data queries (SQL tables)
              - ✉️ Email drafting
              - 📧 Email analysis
              - 💬 General chat

            Ask in natural language, for example:
            - "Search our sustainability report for carbon targets"
            - "Top 10 companies by revenue in the last report"
            - "Draft an email to the CFO about Q4 budget"
            - "Analyze this email: <paste email>"
            """
        ).strip()
        st.session_state.messages = [
            {"role": "assistant", "content": welcome}
        ]


def reset_session() -> None:
    """Clear session state and reinitialize."""
    if "rag_system" in st.session_state:
        try:
            st.session_state.rag_system.close()
        except Exception:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session()


# -----------------------------
# Status / Diagnostics helpers
# -----------------------------
def get_system_status() -> Dict[str, Any]:
    """Return status information similar to display_system_status()."""
    system = st.session_state.rag_system

    try:
        sources = system.retriever.get_available_sources()
    except Exception as e:
        return {"error": f"Could not get sources: {e}"}

    ctx = system.context

    status: Dict[str, Any] = {
        "lmstudio_base_url": LMSTUDIO_BASE_URL,
        "chat_model": system.query_classifier.chat_model,
        "embedding_model": "Snowflake/snowflake-arctic-embed-l",
        "sources_by_type": sources.get("sources_by_type", {}),
        "search_parameters": sources.get("search_parameters", {}),
        "session": {
            "start": ctx.session_start,
            "exchanges": len(ctx.history),
            "last_query_type": ctx.last_query_type.value if ctx.last_query_type else None,
        },
    }
    return status


def render_status_panel() -> None:
    """Render system status in the sidebar."""
    st.sidebar.subheader("🖥️ System Status")

    status = get_system_status()
    if "error" in status:
        st.sidebar.error(status["error"])
        return

    st.sidebar.markdown(
        f"""
        **LM Studio URL:** `{status['lmstudio_base_url']}`  
        **Chat model:** `{status['chat_model']}`  
        **Embeddings:** `{status['embedding_model']}`
        """
    )

    # Session info
    session = status["session"]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Session**")
    st.sidebar.markdown(f"- Started: `{session['start'].strftime('%Y-%m-%d %H:%M:%S')}`")
    st.sidebar.markdown(f"- Exchanges: `{session['exchanges']}`")
    st.sidebar.markdown(
        f"- Last query type: `{session['last_query_type'] or 'None'}`"
    )

    # Sources info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sources**")

    sources_by_type = status.get("sources_by_type", {})
    vector_info = sources_by_type.get("vector", {})
    sql_info = sources_by_type.get("sql", {})

    with st.sidebar.expander("📄 Vector documents", expanded=False):
        st.markdown(f"- Count: `{vector_info.get('count', 0)}`")
        st.markdown(f"- Prefixes: `{vector_info.get('prefixes')}`")
        tables = vector_info.get("tables") or []
        if tables:
            st.markdown("**Sample tables:**")
            for t in tables[:5]:
                rows = t.get("rows")
                rows_str = f"{rows:,} rows" if isinstance(rows, int) else "unknown rows"
                st.markdown(f"- `{t['table_name']}` ({rows_str})")

    with st.sidebar.expander("🗄️ SQL tables", expanded=False):
        st.markdown(f"- Count: `{sql_info.get('count', 0)}`")
        st.markdown(f"- Prefixes: `{sql_info.get('prefixes')}`")
        tables = sql_info.get("tables") or []
        if tables:
            st.markdown("**Sample tables:**")
            for t in tables[:5]:
                rows = t.get("rows")
                rows_str = f"{rows:,} rows" if isinstance(rows, int) else "unknown rows"
                st.markdown(f"- `{t['table_name']}` ({rows_str})")

    # Search configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Search configuration**")
    sp = status.get("search_parameters", {})
    st.sidebar.markdown(
        f"- Vector similarity threshold: `{sp.get('vector_similarity_threshold', 0.7)}`"
    )
    st.sidebar.markdown(
        f"- Max vector results: `{sp.get('max_vector_results', 10)}`"
    )
    st.sidebar.markdown(
        f"- Max SQL results: `{sp.get('max_sql_results', 50)}`"
    )

    # Controls
    st.sidebar.markdown("---")
    if st.sidebar.button("🔁 Reset session", use_container_width=True):
        reset_session()
        st.experimental_rerun()


# -----------------------------
# Main UI
# -----------------------------
def main() -> None:
    st.set_page_config(
        page_title="CEO RAG Chatbot (LM Studio)",
        page_icon="🤖",
        layout="wide",
    )

    init_session()
    render_status_panel()

    st.title("🤖 CEO RAG Chatbot")
    st.caption(
        "LM Studio + Snowflake Arctic embeddings • Document search, data queries, and email assistant"
    )

    # Two-column layout: left = chat, right = info
    col_chat, col_info = st.columns([3, 2])

    # Right column: helper info
    with col_info:
        st.markdown("### ℹ️ How to use")
        st.markdown(
            """
            - **Document search:**  
              *"Search the 2023 annual report for ESG strategy"*
            - **Data query:**  
              *"Show top 10 companies by net sales"*
            - **Email draft:**  
              *"Draft an email to Jane about the new ISO audit findings"*
            - **Email analysis:**  
              *"Analyze this email: \<paste email\>"*
            """
        )
        st.markdown("---")
        st.markdown("### 🧪 Last queries (internal)")
        ctx = st.session_state.rag_system.context
        context_str = ctx.get_context_string(3)
        if context_str.strip():
            st.code(context_str, language="text")
        else:
            st.info("No previous exchanges in this session.")

    # Left column: chat interface
    with col_chat:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a question, or describe what you need...")
        if user_input:
            # Show user message in chat
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            # Process with RAG system
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        system = st.session_state.rag_system
                        response = system.process_query(user_input)
                    except Exception as e:
                        response = f"❌ Unexpected error: {e}"
                    st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )


if __name__ == "__main__":
    main()

