import os
from typing import List, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader

class RAGService:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        """Initialize the RAG service with Ollama configuration."""
        self.base_url = base_url
        self.model_name = model_name
        
        # Initialize Ollama components
        self.embeddings = OllamaEmbeddings(base_url=self.base_url, model=self.model_name)
        self.llm = Ollama(base_url=self.base_url, model=self.model_name)
        
        # Initialize RAG components
        self.vector_store = None
        self.qa_chain = None
        
    def load_documents(self, directory: str = "documents") -> Dict[str, Any]:
        """Load documents from a directory and create embeddings."""
        try:
            # Load documents
            loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            return {"message": f"Successfully loaded {len(documents)} documents"}
        except Exception as e:
            raise Exception(f"Error loading documents: {str(e)}")
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Query the RAG system with the given text."""
        if not self.qa_chain:
            raise Exception("Please load documents first using load_documents()")
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": query_text})
            
            # Get source documents
            docs = self.vector_store.similarity_search(query_text, k=3)
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            return {
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            raise Exception(f"Error querying RAG system: {str(e)}") 