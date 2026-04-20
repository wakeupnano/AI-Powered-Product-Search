"""Application configuration loaded from environment variables.

This module centralizes all system settings using the 12-Factor App methodology.
It ensures that configuration is decoupled from code and type-validated at startup.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Core application settings schema.
    Pydantic automatically maps matching environment variables (from the host or .env)
    to these attributes, enforcing strict type validation.
    """

    # External API & Models
    # Kept empty by default to prevent hardcoding; must be provided in .env
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Database Settings
    # Defined where ChromaDB persists data locally to survive container restarts
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "lg_products"

    # Retrieval and Ranking Tuning
    # How many candidates to fetch from the vector/keyword stores initially
    search_top_k: int = 20
    # Final number of results returned to the user after LLM re-ranking
    final_results: int = 5
    # Smoothing constant for RRF (Standard is 60 per IR literature)
    rrf_k: int = 60

    # Ingestion Setting
    product_catalog_path: str = "./data/lg_products.csv"

    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """
    Singleton factory for application settings.

    Uses @lru_cache so the .env file is parsed and validated exactly once
    at application startup. Subsequent calls return the cached instance from memory,
    avoiding expensive disk I/O operations on every API request.
    """
    return Settings()
