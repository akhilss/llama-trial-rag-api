from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from rag_service import RAGService

app = FastAPI(title="RAG API with Ollama")

# Initialize RAG service
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")
rag_service = RAGService(base_url=OLLAMA_BASE_URL, model_name=MODEL_NAME)

class Query(BaseModel):
    text: str
    collection_name: Optional[str] = "default"

class Response(BaseModel):
    answer: str
    sources: List[str]

@app.post("/load_documents")
async def load_documents(directory: str = "documents"):
    """Load documents from a directory and create embeddings."""
    try:
        result = rag_service.load_documents(directory)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Response)
async def query(query: Query):
    """Query the RAG system."""
    try:
        result = rag_service.query(query.text)
        return Response(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 