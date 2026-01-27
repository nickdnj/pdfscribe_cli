#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) module for PDFScribe
Provides document ingestion, embedding, and semantic search capabilities

Requirements:
- PostgreSQL with pgvector extension
- OpenAI API key for embeddings (text-embedding-3-small)
"""

import os
import hashlib
import logging
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import RealDictCursor
import requests

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension

# Database configuration
DB_HOST = os.getenv("RAG_DB_HOST", "localhost")
DB_PORT = int(os.getenv("RAG_DB_PORT", "5433"))
DB_NAME = os.getenv("RAG_DB_NAME", "rag")
DB_USER = os.getenv("RAG_DB_USER", "rag")
DB_PASSWORD = os.getenv("RAG_DB_PASSWORD", "localdev")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))  # tokens (approximate)
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))  # ~15% overlap


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Represents a search result with its similarity score."""
    chunk_text: str
    source_file: str
    bucket_id: str
    page_number: Optional[int]
    chunk_index: int
    similarity: float
    metadata: Dict[str, Any]


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise


def compute_file_checksum(file_path: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_text_checksum(text: str) -> str:
    """Compute SHA256 checksum of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)."""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Split text into overlapping chunks of approximately chunk_size tokens.
    Tries to split on paragraph or sentence boundaries when possible.
    """
    chunks = []

    # First, split by page breaks if present
    pages = text.split("--- Page Break ---")

    chunk_index = 0
    for page_num, page_text in enumerate(pages, 1):
        page_text = page_text.strip()
        if not page_text:
            continue

        # Split page into paragraphs
        paragraphs = re.split(r'\n\s*\n', page_text)

        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = estimate_tokens(para)

            # If paragraph alone exceeds chunk size, split by sentences
            if para_tokens > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = estimate_tokens(sent)
                    if current_tokens + sent_tokens > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text_content = ' '.join(current_chunk)
                        chunks.append(Chunk(
                            text=chunk_text_content,
                            chunk_index=chunk_index,
                            page_number=page_num if len(pages) > 1 else None
                        ))
                        chunk_index += 1

                        # Start new chunk with overlap
                        overlap_tokens = 0
                        overlap_content = []
                        for item in reversed(current_chunk):
                            item_tokens = estimate_tokens(item)
                            if overlap_tokens + item_tokens <= overlap:
                                overlap_content.insert(0, item)
                                overlap_tokens += item_tokens
                            else:
                                break
                        current_chunk = overlap_content
                        current_tokens = overlap_tokens

                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            else:
                # Check if adding paragraph exceeds chunk size
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text_content = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text_content,
                        chunk_index=chunk_index,
                        page_number=page_num if len(pages) > 1 else None
                    ))
                    chunk_index += 1

                    # Start new chunk with overlap
                    overlap_tokens = 0
                    overlap_content = []
                    for item in reversed(current_chunk):
                        item_tokens = estimate_tokens(item)
                        if overlap_tokens + item_tokens <= overlap:
                            overlap_content.insert(0, item)
                            overlap_tokens += item_tokens
                        else:
                            break
                    current_chunk = overlap_content
                    current_tokens = overlap_tokens

                current_chunk.append(para)
                current_tokens += para_tokens

        # Don't forget the last chunk of the page
        if current_chunk:
            chunk_text_content = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text_content,
                chunk_index=chunk_index,
                page_number=page_num if len(pages) > 1 else None
            ))
            chunk_index += 1

    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using OpenAI's embedding API.
    Handles batching automatically (max 2048 texts per request).
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required for embeddings")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    all_embeddings = []
    batch_size = 100  # Process in smaller batches to avoid timeouts

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch
        }

        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            error_info = response.json() if response.text else {"error": response.status_code}
            raise RuntimeError(f"Embedding API error: {error_info}")

        result = response.json()
        batch_embeddings = [item['embedding'] for item in result['data']]
        all_embeddings.extend(batch_embeddings)

        logging.debug(f"Got embeddings for batch {i // batch_size + 1}")

    return all_embeddings


def is_document_indexed(bucket_id: str, source_file: str, checksum: str) -> bool:
    """Check if a document with the given checksum is already indexed."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM indexed_documents
                WHERE bucket_id = %s AND source_file = %s AND file_checksum = %s
            """, (bucket_id, source_file, checksum))
            return cur.fetchone() is not None
    finally:
        conn.close()


def ingest_document(
    text: str,
    bucket_id: str,
    source_file: str,
    file_checksum: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Ingest a document into the RAG system.

    Args:
        text: The document text to ingest
        bucket_id: Context bucket ID (e.g., 'wharfside-docs')
        source_file: Original file path or name
        file_checksum: SHA256 checksum of the source file (computed if not provided)
        metadata: Additional metadata to store with chunks
        force: Re-index even if document is already indexed

    Returns:
        Dict with ingestion results (chunk_count, total_tokens, etc.)
    """
    if not file_checksum:
        file_checksum = compute_text_checksum(text)

    # Check if already indexed
    if not force and is_document_indexed(bucket_id, source_file, file_checksum):
        logging.info(f"Document already indexed: {source_file}")
        return {"status": "skipped", "reason": "already_indexed", "checksum": file_checksum}

    # Chunk the document
    chunks = chunk_text(text)
    if not chunks:
        return {"status": "error", "reason": "no_chunks_created"}

    logging.info(f"Created {len(chunks)} chunks from {source_file}")

    # Get embeddings for all chunks
    chunk_texts = [c.text for c in chunks]
    embeddings = get_embeddings(chunk_texts)

    logging.info(f"Generated embeddings for {len(embeddings)} chunks")

    # Store in database
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Delete existing entries for this document if force re-indexing
            if force:
                cur.execute("""
                    DELETE FROM embeddings
                    WHERE bucket_id = %s AND source_file = %s
                """, (bucket_id, source_file))
                cur.execute("""
                    DELETE FROM indexed_documents
                    WHERE bucket_id = %s AND source_file = %s
                """, (bucket_id, source_file))

            # Insert chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk_metadata = metadata.copy() if metadata else {}
                cur.execute("""
                    INSERT INTO embeddings
                    (bucket_id, source_file, page_number, chunk_index, chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    bucket_id,
                    source_file,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                    embedding,
                    json.dumps(chunk_metadata)
                ))

            # Record document as indexed
            total_tokens = sum(estimate_tokens(c.text) for c in chunks)
            cur.execute("""
                INSERT INTO indexed_documents
                (bucket_id, source_file, file_checksum, chunk_count, total_tokens, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (bucket_id, source_file)
                DO UPDATE SET
                    file_checksum = EXCLUDED.file_checksum,
                    chunk_count = EXCLUDED.chunk_count,
                    total_tokens = EXCLUDED.total_tokens,
                    metadata = EXCLUDED.metadata,
                    indexed_at = NOW()
            """, (
                bucket_id,
                source_file,
                file_checksum,
                len(chunks),
                total_tokens,
                json.dumps(metadata or {})
            ))

            conn.commit()
    finally:
        conn.close()

    return {
        "status": "success",
        "chunk_count": len(chunks),
        "total_tokens": total_tokens,
        "checksum": file_checksum,
        "source_file": source_file,
        "bucket_id": bucket_id
    }


def search_documents(
    query: str,
    bucket_id: Optional[str] = None,
    limit: int = 10,
    similarity_threshold: float = 0.5
) -> List[SearchResult]:
    """
    Perform semantic search across indexed documents.

    Args:
        query: Search query text
        bucket_id: Optional filter by context bucket
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of SearchResult objects sorted by similarity (descending)
    """
    # Get embedding for query
    query_embedding = get_embeddings([query])[0]

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build query with optional bucket filter
            if bucket_id:
                cur.execute("""
                    SELECT
                        chunk_text,
                        source_file,
                        bucket_id,
                        page_number,
                        chunk_index,
                        1 - (embedding <=> %s::vector) as similarity,
                        metadata
                    FROM embeddings
                    WHERE bucket_id = %s
                    AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, bucket_id, query_embedding, similarity_threshold, query_embedding, limit))
            else:
                cur.execute("""
                    SELECT
                        chunk_text,
                        source_file,
                        bucket_id,
                        page_number,
                        chunk_index,
                        1 - (embedding <=> %s::vector) as similarity,
                        metadata
                    FROM embeddings
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, similarity_threshold, query_embedding, limit))

            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append(SearchResult(
                    chunk_text=row['chunk_text'],
                    source_file=row['source_file'],
                    bucket_id=row['bucket_id'],
                    page_number=row['page_number'],
                    chunk_index=row['chunk_index'],
                    similarity=float(row['similarity']),
                    metadata=row['metadata'] if row['metadata'] else {}
                ))

            return results
    finally:
        conn.close()


def list_indexed_documents(bucket_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all indexed documents.

    Args:
        bucket_id: Optional filter by context bucket

    Returns:
        List of document info dicts
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if bucket_id:
                cur.execute("""
                    SELECT bucket_id, source_file, file_checksum, chunk_count,
                           total_tokens, indexed_at, metadata
                    FROM indexed_documents
                    WHERE bucket_id = %s
                    ORDER BY indexed_at DESC
                """, (bucket_id,))
            else:
                cur.execute("""
                    SELECT bucket_id, source_file, file_checksum, chunk_count,
                           total_tokens, indexed_at, metadata
                    FROM indexed_documents
                    ORDER BY bucket_id, indexed_at DESC
                """)

            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_document(bucket_id: str, source_file: str) -> bool:
    """
    Delete a document and its embeddings from the index.

    Returns:
        True if document was deleted, False if not found
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM embeddings
                WHERE bucket_id = %s AND source_file = %s
            """, (bucket_id, source_file))

            cur.execute("""
                DELETE FROM indexed_documents
                WHERE bucket_id = %s AND source_file = %s
            """, (bucket_id, source_file))

            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    finally:
        conn.close()


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the RAG index."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total documents
            cur.execute("SELECT COUNT(*) as doc_count FROM indexed_documents")
            doc_count = cur.fetchone()['doc_count']

            # Total chunks
            cur.execute("SELECT COUNT(*) as chunk_count FROM embeddings")
            chunk_count = cur.fetchone()['chunk_count']

            # Total tokens
            cur.execute("SELECT COALESCE(SUM(total_tokens), 0) as total_tokens FROM indexed_documents")
            total_tokens = cur.fetchone()['total_tokens']

            # Documents by bucket
            cur.execute("""
                SELECT bucket_id, COUNT(*) as count
                FROM indexed_documents
                GROUP BY bucket_id
            """)
            by_bucket = {row['bucket_id']: row['count'] for row in cur.fetchall()}

            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "total_tokens": total_tokens,
                "documents_by_bucket": by_bucket
            }
    finally:
        conn.close()


# CLI interface when run directly
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="RAG operations for PDFScribe")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", help="Text file to ingest")
    ingest_parser.add_argument("--bucket", required=True, help="Context bucket ID")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--bucket", help="Filter by context bucket")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")

    # List command
    list_parser = subparsers.add_parser("list", help="List indexed documents")
    list_parser.add_argument("--bucket", help="Filter by context bucket")

    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document from index")
    delete_parser.add_argument("--bucket", required=True, help="Context bucket ID")
    delete_parser.add_argument("--file", required=True, help="Source file name")

    args = parser.parse_args()

    if args.command == "ingest":
        with open(args.file, 'r') as f:
            text = f.read()
        result = ingest_document(
            text=text,
            bucket_id=args.bucket,
            source_file=os.path.basename(args.file),
            force=args.force
        )
        print(json.dumps(result, indent=2))

    elif args.command == "search":
        results = search_documents(
            query=args.query,
            bucket_id=args.bucket,
            limit=args.limit
        )
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (similarity: {r.similarity:.3f}) ---")
            print(f"Source: {r.source_file} (bucket: {r.bucket_id})")
            if r.page_number:
                print(f"Page: {r.page_number}")
            print(f"Text: {r.chunk_text[:200]}...")

    elif args.command == "list":
        docs = list_indexed_documents(args.bucket)
        for doc in docs:
            print(f"{doc['bucket_id']}/{doc['source_file']}: "
                  f"{doc['chunk_count']} chunks, {doc['total_tokens']} tokens, "
                  f"indexed {doc['indexed_at']}")

    elif args.command == "stats":
        stats = get_index_stats()
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "delete":
        deleted = delete_document(args.bucket, args.file)
        print(f"Deleted: {deleted}")

    else:
        parser.print_help()
