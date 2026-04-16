# LG Product Semantic Search

An AI-powered search system that lets users find LG products using natural language queries with a sample product catalog CSV. The system understands intent, extracts filters, and explains why each result matches. 

```
## Features

- **Semantic Retrieval:** Uses high-dimensional text embeddings to understand the contextual meaning of queries.
- **Intelligent Query Parsing:** Leverages an LLM as a "Query Router" to dynamically extract hard filters (like price maximums and product categories) from natural language.
- **Hybrid Search Capabilities:** Combines metadata filtering with vector similarity to guarantee accurate results (e.g., strictly enforcing a budget while finding the most semantically relevant items).
- **Explainable AI:** Generates short, dynamic explanations for *why* a specific product matches the user's query.
- **Production-Ready:** Containerized with Docker, complete with input validation (Pydantic), error handling, and a health-check endpoint.

## Architecture

The search pipeline is designed around a multi-stage retrieval-augmented generation (RAG) and search philosophy:

1. **Data Ingestion & Preprocessing:** 
2. **Embedding & Storage:** 
3. **Query Understanding:** 
4. **Vector Retrieval:** 
5. **Result Generation:** 

## Tech Stack

- **Python 3.11+**
- **FastAPI** — async web framework with auto-generated OpenAPI docs
- **ChromaDB** — vector database with metadata filtering