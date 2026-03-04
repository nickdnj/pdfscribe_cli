"""
RAG REST API — FastAPI service wrapping the RAG module.
Deployed on Google Cloud Run, connects to Cloud SQL (pgvector).

Endpoints:
    POST /v1/search       — Semantic search
    POST /v1/ingest       — Ingest text
    POST /v1/ingest-file  — Ingest uploaded file
    GET  /v1/stats        — Index statistics
    GET  /v1/documents    — List indexed documents
    DELETE /v1/documents  — Delete a document
    GET  /v1/health       — Health check
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

# Add parent src to path so we can import rag module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Force postgres backend for the API
os.environ.setdefault("RAG_BACKEND", "postgres")

from rag import (
    search_documents, ingest_document, list_indexed_documents,
    delete_document, get_index_stats
)
from api.auth import verify_api_key

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAG Shared Brain API",
    version="1.0.0",
    description="Semantic search and document ingestion for the Agent Architect ecosystem",
)


# --- Request/Response Models ---

class SearchRequest(BaseModel):
    query: str
    bucket_id: Optional[str] = None
    limit: int = 10
    similarity_threshold: float = 0.35


class SearchResultItem(BaseModel):
    chunk_text: str
    source_file: str
    bucket_id: str
    page_number: Optional[int]
    chunk_index: int
    similarity: float
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    count: int


class IngestRequest(BaseModel):
    text: str
    bucket_id: str
    source_file: str
    force: bool = False
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    status: str
    chunk_count: int = 0
    total_tokens: int = 0
    source_file: str = ""
    bucket_id: str = ""


class DeleteRequest(BaseModel):
    bucket_id: str
    source_file: str


# --- Endpoints ---

@app.get("/v1/health")
async def health():
    """Health check — no auth required."""
    return {"status": "ok", "backend": os.getenv("RAG_BACKEND", "unknown")}


@app.post("/v1/search", response_model=SearchResponse)
async def search(req: SearchRequest, _: str = Depends(verify_api_key)):
    """Perform semantic search across indexed documents."""
    try:
        results = search_documents(
            query=req.query,
            bucket_id=req.bucket_id,
            limit=req.limit,
            similarity_threshold=req.similarity_threshold,
        )
        return SearchResponse(
            results=[
                SearchResultItem(
                    chunk_text=r.chunk_text,
                    source_file=r.source_file,
                    bucket_id=r.bucket_id,
                    page_number=r.page_number,
                    chunk_index=r.chunk_index,
                    similarity=r.similarity,
                    metadata=r.metadata,
                )
                for r in results
            ],
            query=req.query,
            count=len(results),
        )
    except Exception as e:
        logging.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, _: str = Depends(verify_api_key)):
    """Ingest text into the RAG index."""
    try:
        result = ingest_document(
            text=req.text,
            bucket_id=req.bucket_id,
            source_file=req.source_file,
            metadata=req.metadata,
            force=req.force,
        )
        return IngestResponse(
            status=result.get("status", "unknown"),
            chunk_count=result.get("chunk_count", 0),
            total_tokens=result.get("total_tokens", 0),
            source_file=result.get("source_file", req.source_file),
            bucket_id=result.get("bucket_id", req.bucket_id),
        )
    except Exception as e:
        logging.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest-file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    bucket_id: str = Form(...),
    force: bool = Form(False),
    _: str = Depends(verify_api_key),
):
    """Ingest an uploaded file into the RAG index."""
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")
        result = ingest_document(
            text=text,
            bucket_id=bucket_id,
            source_file=file.filename or "uploaded-file",
            force=force,
        )
        return IngestResponse(
            status=result.get("status", "unknown"),
            chunk_count=result.get("chunk_count", 0),
            total_tokens=result.get("total_tokens", 0),
            source_file=result.get("source_file", file.filename),
            bucket_id=result.get("bucket_id", bucket_id),
        )
    except Exception as e:
        logging.exception("File ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats")
async def stats(_: str = Depends(verify_api_key)):
    """Get RAG index statistics."""
    try:
        return get_index_stats()
    except Exception as e:
        logging.exception("Stats failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents")
async def documents(bucket_id: Optional[str] = None, _: str = Depends(verify_api_key)):
    """List indexed documents."""
    try:
        return list_indexed_documents(bucket_id)
    except Exception as e:
        logging.exception("List documents failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/documents")
async def delete_doc(req: DeleteRequest, _: str = Depends(verify_api_key)):
    """Delete a document from the index."""
    try:
        deleted = delete_document(req.bucket_id, req.source_file)
        return {"deleted": deleted}
    except Exception as e:
        logging.exception("Delete failed")
        raise HTTPException(status_code=500, detail=str(e))
