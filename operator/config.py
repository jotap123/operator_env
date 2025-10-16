import os
from datetime import datetime
from dotenv import load_dotenv

from pydantic_settings import BaseSettings
from sentence_transformers.cross_encoder import CrossEncoder

TODAY = datetime.today().date().strftime("%Y-%m-%d")
load_dotenv()


class AgentSettings(BaseSettings):
    """
    Centralized configuration for the RAG agent using Pydantic.
    Allows for easy management of settings and loading from environment variables.
    """
    # Tavily Search API
    TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

    # Document Chunking Strategy
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Hybrid Search & Retrieval Parameters
    MAX_SEARCH_RESULTS: int = 20
    VECTOR_K: int = 20
    BM25_K: int = 20
    RERANK_TOP_K: int = 5
    
    # Ensemble Retriever Weights (Vector vs. BM25)
    ENSEMBLE_WEIGHTS: list[float] = [0.3, 0.7]

    # Memory Management
    MEMORY_SUMMARIZER_THRESHOLD: int = 6 # Number of messages before summarizing

    # Initialize the model directly from Hugging Face
    CROSS_ENCODER = CrossEncoder('BAAI/bge-reranker-large', max_length=512)

    class Config:
        # Pydantic configuration
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

# Create a single instance to be imported across the application
settings = AgentSettings()
