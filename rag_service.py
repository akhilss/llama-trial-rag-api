import logging
import os
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
import glob
import pandas as pd

class DocumentManager:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        """Initialize the document manager with Ollama configuration."""
        self.base_url = base_url
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(base_url=self.base_url, model=self.model_name)
        self.vector_store = None
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(f"DocumentManager initialized with base_url: {self.base_url} and model_name: {self.model_name}")

    def load_documents(self, directory: str = "documents") -> Dict[str, Any]:
        """Load documents from a directory and create embeddings."""
        try:
            # Load text documents
            text_loader = DirectoryLoader(
                directory,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            text_documents = text_loader.load()
            
            # Load CSV documents
            csv_loader = DirectoryLoader(
                directory,
                glob="**/*.csv",
                loader_cls=CSVLoader
            )
            csv_documents = csv_loader.load()

            # Combine all documents
            all_documents = text_documents + csv_documents
            
            if not all_documents:
                raise Exception(f"No documents found in {directory}")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_documents)
            
            # Create vector store in memory
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            return {
                "message": f"Successfully loaded {len(text_documents)} text documents and {len(csv_documents)} CSV documents"
            }
        except Exception as e:
            logging.exception("Error loading documents")
            raise Exception(f"Error loading documents: {str(e)}")

    def clear_documents(self) -> Dict[str, str]:
        """Clear the vector store."""
        self.vector_store = None
        return {"message": "Vector store cleared successfully"}

class RAGService:
    def __init__(self, document_manager: DocumentManager, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        """Initialize the RAG service with document manager and Ollama configuration."""
        self.document_manager = document_manager
        self.base_url = base_url
        self.model_name = model_name
        self.llm = OllamaLLM(base_url=self.base_url, model=self.model_name)
        self.qa_chain = None

    def _initialize_qa_chain(self):
        """Initialize the QA chain with the current vector store."""
        if not self.document_manager.vector_store:
            raise Exception("Please load documents first using document_manager.load_documents()")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.document_manager.vector_store.as_retriever(search_kwargs={"k": 3})
        )

    def query(self, query_text: str) -> Dict[str, Any]:
        """Query the RAG system with the given text."""
        if not self.qa_chain:
            self._initialize_qa_chain()
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": query_text})
            
            # Get source documents
            docs = self.document_manager.vector_store.similarity_search(query_text, k=3)
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            return {
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            raise Exception(f"Error querying RAG system: {str(e)}") 