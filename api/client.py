"""
Thin Python client for the RAG REST API.
Used by remote/exported agents to access the shared brain.

Usage:
    from rag_client import RAGClient
    client = RAGClient(url="https://rag-api-xxx.run.app", api_key=os.getenv("RAG_API_KEY"))
    results = client.search("parking rules", bucket_id="wharfside-docs")
    client.ingest(text="...", bucket_id="session-logs", source_file="discovery.md")
"""

import os
from typing import Optional

import requests


class RAGClient:
    """Client for the RAG REST API."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = (url or os.getenv("RAG_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("RAG_API_KEY", "")
        if not self.url:
            raise ValueError("RAG_API_URL or url parameter required")
        if not self.api_key:
            raise ValueError("RAG_API_KEY or api_key parameter required")

    def _headers(self):
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def search(self, query: str, bucket_id: Optional[str] = None,
               limit: int = 10, threshold: float = 0.35) -> list:
        """Semantic search across indexed documents."""
        resp = requests.post(
            f"{self.url}/v1/search",
            headers=self._headers(),
            json={
                "query": query,
                "bucket_id": bucket_id,
                "limit": limit,
                "similarity_threshold": threshold,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def ingest(self, text: str, bucket_id: str, source_file: str,
               force: bool = False, metadata: Optional[dict] = None) -> dict:
        """Ingest text into the RAG index."""
        resp = requests.post(
            f"{self.url}/v1/ingest",
            headers=self._headers(),
            json={
                "text": text,
                "bucket_id": bucket_id,
                "source_file": source_file,
                "force": force,
                "metadata": metadata,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> dict:
        """Get RAG index statistics."""
        resp = requests.get(
            f"{self.url}/v1/stats",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def documents(self, bucket_id: Optional[str] = None) -> list:
        """List indexed documents."""
        params = {"bucket_id": bucket_id} if bucket_id else {}
        resp = requests.get(
            f"{self.url}/v1/documents",
            headers=self._headers(),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self, bucket_id: str, source_file: str) -> dict:
        """Delete a document from the index."""
        resp = requests.delete(
            f"{self.url}/v1/documents",
            headers=self._headers(),
            json={"bucket_id": bucket_id, "source_file": source_file},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Health check (no auth required)."""
        resp = requests.get(f"{self.url}/v1/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
