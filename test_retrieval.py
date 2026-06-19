# test_retrieval.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_store", embedding_function=embeddings)

results = db.similarity_search("sick leave", k=4)
for i, doc in enumerate(results):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:300])