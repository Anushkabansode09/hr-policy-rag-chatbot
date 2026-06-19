"""
api.py
------
FastAPI backend for HR Policy Chatbot.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import build_chain, query_chain

app = FastAPI(
    title="HR Policy Chatbot API",
    description="RAG-based HR policy Q&A using ChromaDB + Groq LLM",
    version="1.0.0",
)

# ── Load chain once at startup ────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    app.state.chain = build_chain()

# ── Request/Response models ───────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "HR Policy Chatbot API"}

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    result = query_chain(app.state.chain, request.question)
    
    sources = [
        {
            "content": doc.page_content[:300],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0) + 1,
        }
        for doc in result["sources"]
    ]
    
    return AnswerResponse(answer=result["answer"], sources=sources)

@app.get("/health")
def health():
    return {"status": "healthy"}