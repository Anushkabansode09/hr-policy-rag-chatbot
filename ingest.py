"""
ingest.py
---------
Handles: PDF loading → text chunking → embedding → storing in ChromaDB

Run this once per document (or when you add new documents).
Usage: python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR = "sample_docs"          # folder where your HR PDFs live
CHROMA_DIR = "chroma_store"       # where ChromaDB will persist data on disk
EMBED_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model (downloads once, ~90MB)

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between consecutive chunks (preserves context)

# ── Step 1: Load all PDFs from the docs folder ───────────────────────────────

def load_documents(docs_dir: str):
    """Load all PDF files from docs_dir. Returns list of LangChain Document objects."""
    documents = []

    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Folder '{docs_dir}' not found. Create it and add PDF files.")

    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in '{docs_dir}'.")

    for pdf_file in pdf_files:
        path = os.path.join(docs_dir, pdf_file)
        print(f"  Loading: {pdf_file}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
        print(f"    → {len(docs)} pages loaded")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


# ── Step 2: Split documents into chunks ──────────────────────────────────────

def split_documents(documents):
    """
    Split pages into smaller overlapping chunks.
    RecursiveCharacterTextSplitter tries to split on paragraphs → sentences → words
    in that order, so chunks are semantically cleaner than hard character cuts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # priority order for splitting
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# ── Step 3: Embed + store in ChromaDB ────────────────────────────────────────

def store_in_chroma(chunks):
    """
    Convert each chunk to a vector using HuggingFace sentence-transformers,
    then store all vectors + original text in ChromaDB on disk.
    """
    print(f"\nLoading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding chunks and storing in ChromaDB ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nDone. {len(chunks)} chunks stored in '{CHROMA_DIR}/'")
    print("ChromaDB is ready for querying.")
    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("HR Policy Chatbot — Document Ingestion")
    print("=" * 50)

    print(f"\nStep 1: Loading PDFs from '{DOCS_DIR}/'")
    documents = load_documents(DOCS_DIR)

    print(f"\nStep 2: Chunking documents")
    chunks = split_documents(documents)

    print(f"\nStep 3: Embedding + storing in ChromaDB")
    store_in_chroma(chunks)

    print("\nIngestion complete. You can now run app.py")
