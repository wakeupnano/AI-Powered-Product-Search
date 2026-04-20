"""Pydantic models for API request/response schemas.

These models serve as the strict data contracts for the FastAPI endpoints.
They handle automatic validation, serialization, and generate the OpenAPI/Swagger documentation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    Validates incoming POST requests to the /search endpoint.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500, # Prevents malicious payloads designed to drain API tokens
        description="Natural language search query",
        examples=["energy efficient large refrigerator under $2000"],
    )


class ParsedQuery(BaseModel):
    """
    Internal model used to map and validate the structured JSON output
    returned by the LLM Query Router.
    """
    semantic_query: str
    # Optional fields map to the hard metadata filters applied during ChromaDB retrieval
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


class SearchResult(BaseModel):
    """
    Represents a single LG product match formatted for the client response.
    """
    product_name: str
    sku: str
    price: float
    relevance_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class SearchResponse(BaseModel):
    """
    The final payload returned to the client upon a successful search.
    Includes diagnostic info (parsed_query) alongside the actual results.
    """
    query: str
    parsed_query: ParsedQuery
    results: list[SearchResult]
    total_results: int


class HealthResponse(BaseModel):
    """
    Schema for the /health endpoint to verify system operational status.
    """
    status: str = "healthy"
    product_count: int
    index_loaded: bool
