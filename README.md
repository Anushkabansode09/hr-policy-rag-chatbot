# HR Policy Chatbot 🤖📋

A Retrieval-Augmented Generation (RAG) chatbot that answers employee questions based on company HR policy documents.

Built with LangChain, ChromaDB, Groq (Llama 3), and Streamlit.

---

## How It Works

```
PDF Documents
    ↓ PyPDFLoader
Raw Text (pages)
    ↓ RecursiveCharacterTextSplitter
Chunks (500 chars, 50 overlap)
    ↓ sentence-transformers (all-MiniLM-L6-v2)
Vectors
    ↓ ChromaDB (persisted on disk)

At query time:
User Question → embed → ChromaDB similarity search → top 4 chunks
→ Groq (Llama 3) → grounded answer + source attribution
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3 8B via Groq API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (local persistent) |
| Framework | LangChain |
| UI | Streamlit |

---

## Setup

### 1. Clone and install dependencies
```bash
git clone https://github.com/Anushkabansode09/hr-policy-chatbot
cd hr-policy-chatbot
pip install -r requirements.txt
```

### 2. Add your Groq API key
```bash
# Edit .env file
GROQ_API_KEY=your_key_here
```
Get a free key at: https://console.groq.com

### 3. Add HR policy PDFs
```bash
mkdir sample_docs
# copy your PDF files into sample_docs/
```

### 4. Run ingestion (one time per document)
```bash
python ingest.py
```

### 5. Launch the app
```bash
streamlit run app.py
```

---

## Project Structure

```
hr-policy-chatbot/
├── app.py               # Streamlit UI
├── rag_pipeline.py      # RAG chain (retriever + LLM + memory)
├── ingest.py            # PDF → chunks → ChromaDB
├── requirements.txt
├── .env                 # GROQ_API_KEY (not committed)
├── .gitignore
└── sample_docs/         # Add HR PDF files here
```

---

## Features

- Multi-turn conversation with memory
- Source attribution (shows which document chunks were used)
- Persistent vector store (no re-ingestion needed on restart)
- Handles multiple PDF documents
- Graceful error handling

---

## Author

Anushka Bansode — [github.com/Anushkabansode09](https://github.com/Anushkabansode09)
