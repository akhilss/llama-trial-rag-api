# RAG API with Ollama

This is a FastAPI-based RAG (Retrieval Augmented Generation) system that uses Ollama as the LLM backend. The system allows you to load documents, create embeddings, and query them using natural language.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally (or accessible via network)
- A model pulled in Ollama (default: llama2)

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `documents` directory and add your text files:
```bash
mkdir documents
# Add your .txt files to the documents directory
```

## Running the API

Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Load Documents
```bash
POST /load_documents
```
Loads documents from the specified directory and creates embeddings.

### 2. Query
```bash
POST /query
```
Query the RAG system with natural language.

Example request:
```json
{
    "text": "What is the main topic of the documents?",
    "collection_name": "default"
}
```

Example response:
```json
{
    "answer": "The main topic is...",
    "sources": ["documents/file1.txt", "documents/file2.txt"]
}
```

## Environment Variables

- `OLLAMA_BASE_URL`: URL of the Ollama server (default: http://localhost:11434)
- `MODEL_NAME`: Name of the Ollama model to use (default: llama2)

## Notes

- The system uses ChromaDB for vector storage
- Documents are split into chunks of 1000 characters with 200 character overlap
- The system retrieves the top 3 most relevant documents for each query 