from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.rag_pipeline import RAGPipeline
import os

# Initialize FastAPI app
app = FastAPI(
    title="Lexi Legal RAG Backend",
    description="A Retrieval-Augmented Generation (RAG) service for legal queries.",
    version="1.0.0",
)


# Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str


# Pydantic model for the citation format in the response
class Citation(BaseModel):
    text: str
    source: str


# Pydantic model for the response format
class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]


# Initialize the RAG pipeline globally to load the vector store once
# This ensures the FAISS index and metadata are loaded when the app starts
rag_pipeline = RAGPipeline()

# Check if the RAG pipeline loaded successfully
if rag_pipeline.faiss_index is None or rag_pipeline.metadata is None:
    print(
        "WARNING: RAG pipeline failed to load vector store. Ensure document_processor.py was run."
    )
    print(
        "The API will still start, but queries will return an error until the vector store is available."
    )


@app.post("/query", response_model=QueryResponse)
async def query_legal_documents(request: QueryRequest):
    """
    Processes a natural language legal query using RAG and returns a generated answer
    with relevant citations.
    """
    if rag_pipeline.faiss_index is None or rag_pipeline.metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: Document vector store is not loaded. Please run document_processor.py first.",
        )

    try:
        # Call the RAG pipeline to get the answer and citations
        result = rag_pipeline.query_rag_pipeline(request.query)

        # Return the formatted response
        return QueryResponse(
            answer=result.get("answer", "No answer generated."),
            citations=[
                Citation(text=c["text"], source=c["source"])
                for c in result.get("citations", [])
            ],
        )
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/")
async def root():
    """
    Root endpoint for basic health check.
    """
    return {
        "message": "Lexi Legal RAG Backend is running. Use /query endpoint for legal queries."
    }


# To run this application, use Uvicorn:
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# The --reload flag is useful for development as it restarts the server on code changes.
