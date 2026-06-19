"""
rag_pipeline.py
---------------
Handles: loading ChromaDB → retrieving relevant chunks → calling Groq LLM → returning answer

This is the brain of the chatbot. Imported by app.py.
"""

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"    # fast, free tier — can upgrade to llama3-70b
TOP_K = 4                          # number of chunks to retrieve per query

# ── Prompt Template ───────────────────────────────────────────────────────────
# Tells the LLM exactly how to behave — grounded in context, no hallucination

SYSTEM_PROMPT = """You are an HR Policy Assistant. Your job is to answer employee questions 
accurately based ONLY on the provided HR policy documents.

Rules:
- Answer only from the context provided below.
- If the answer is not in the context, say: "I couldn't find this in the HR policy documents."
- Be concise and direct. Use bullet points where appropriate.
- Always mention which section or policy the answer comes from if visible in context.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT,
)

# ── Load vectorstore ──────────────────────────────────────────────────────────

def load_vectorstore():
    """Load the persisted ChromaDB vectorstore from disk."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB store not found at '{CHROMA_DIR}/'. Run ingest.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


# ── Build the RAG chain ───────────────────────────────────────────────────────

def build_chain():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=groq_api_key,
        temperature=0.2,
        max_tokens=1024,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,  # add this to see what's happening
    )

    return chain
# ── Query function ────────────────────────────────────────────────────────────

def query_chain(chain, question: str) -> dict:
    """
    Send a question to the chain. Returns dict with:
      - answer: str
      - source_documents: list of Document objects (chunk text + metadata)
    """
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result["source_documents"],
    }
