"""Pydantic models for API request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language search query",
        examples=["energy efficient large refrigerator under $2000"],
    )


class ParsedQuery(BaseModel):
    semantic_query: str
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


class SearchResult(BaseModel):
    product_name: str
    sku: str
    price: float
    relevance_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class SearchResponse(BaseModel):
    query: str
    parsed_query: ParsedQuery
    results: list[SearchResult]
    total_results: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    product_count: int
    index_loaded: bool
