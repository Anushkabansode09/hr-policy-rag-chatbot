"""
app.py
------
Streamlit UI for the HR Policy Chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
from rag_pipeline import build_chain, query_chain

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="📋",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📋 HR Policy Chatbot")
st.caption("Ask questions about company policies. Answers are grounded in uploaded HR documents.")
st.divider()

# ── Load chain (cached so it only loads once per session) ─────────────────────

@st.cache_resource(show_spinner="Loading HR documents and AI model...")
def get_chain():
    """Build the RAG chain once and cache it for the session."""
    try:
        return build_chain()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()

chain = get_chain()

# ── Session state: store chat history for display ─────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display existing chat messages ────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if they exist on an assistant message
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Source chunks used"):
                for i, doc in enumerate(message["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "document")
                    st.markdown(f"**Chunk {i}** — {source}, page {page + 1}")
                    st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                    st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about leave policy, code of conduct, benefits..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents..."):
            result = query_chain(chain, prompt)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)

        # Show source chunks in expandable section
        if sources:
            with st.expander("📄 Source chunks used"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "document")
                    st.markdown(f"**Chunk {i}** — {source}, page {page + 1}")
                    st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                    st.divider()

    # Save assistant message with sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot answers questions based on company HR policy documents.
    
    **Powered by:**
    - 🦙 Llama 3 via Groq
    - 🔍 ChromaDB vector search
    - 🔗 LangChain
    - 🤗 sentence-transformers
    """)

    st.divider()

    st.header("💡 Sample Questions")
    st.markdown("""
    - What is the leave policy?
    - How many sick days am I allowed?
    - What is the code of conduct?
    - How do I apply for work from home?
    - What are the performance review criteria?
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
