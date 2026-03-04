"""API key authentication for the RAG REST API."""

import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# API keys can be comma-separated for multiple keys
_RAG_API_KEYS = os.getenv("RAG_API_KEYS", "")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> set:
    """Parse API keys from environment."""
    if not _RAG_API_KEYS:
        return set()
    return {k.strip() for k in _RAG_API_KEYS.split(",") if k.strip()}


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the request header."""
    valid_keys = get_api_keys()
    if not valid_keys:
        # No keys configured — reject all requests
        raise HTTPException(status_code=500, detail="No API keys configured")
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
