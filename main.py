import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from rag_service import RAGService, DocumentManager

app = FastAPI(title="RAG API with Ollama")

# Initialize services
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")
document_manager = DocumentManager(base_url=OLLAMA_BASE_URL, model_name=MODEL_NAME)
rag_service = RAGService(document_manager=document_manager, base_url=OLLAMA_BASE_URL, model_name=MODEL_NAME)
logging.basicConfig(level=logging.DEBUG)

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
        result = document_manager.load_documents(directory)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_documents")
async def clear_documents():
    """Clear the vector store."""
    try:
        result = document_manager.clear_documents()
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