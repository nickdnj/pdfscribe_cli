#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) module for PDFScribe
Provides document ingestion, embedding, and semantic search capabilities

Supports two backends (switched via RAG_BACKEND env var):
- "sqlite"   — Local SQLite + sqlite-vec (default, original)
- "postgres" — Cloud PostgreSQL + pgvector

Requirements:
- SQLite backend: sqlite-vec extension
- Postgres backend: psycopg2-binary, pgvector-enabled Postgres
- Both: OpenAI API key for embeddings (text-embedding-3-small)
"""

import os
import hashlib
import logging
import json
import re
import struct
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import requests

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension

# Backend selection
RAG_BACKEND = os.getenv("RAG_BACKEND", "sqlite")  # "sqlite" or "postgres"

# SQLite configuration
DB_PATH = os.getenv("RAG_DB_PATH", os.path.join(os.path.expanduser("~"), ".wharfside", "rag.db"))

# Postgres configuration
PG_HOST = os.getenv("RAG_DB_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("RAG_DB_PORT", "5433"))
PG_USER = os.getenv("RAG_DB_USER", "rag")
PG_PASSWORD = os.getenv("RAG_DB_PASSWORD", "")
PG_DATABASE = os.getenv("RAG_DB_NAME", "rag")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))  # tokens (approximate)
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))  # ~15% overlap


def _is_postgres():
    return RAG_BACKEND == "postgres"


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


# ---------------------------------------------------------------------------
# Embedding serialization (SQLite only — Postgres uses vector type natively)
# ---------------------------------------------------------------------------

def serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize a float list to bytes for sqlite-vec."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(blob: bytes) -> List[float]:
    """Deserialize bytes to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


# ---------------------------------------------------------------------------
# SQLite backend helpers
# ---------------------------------------------------------------------------

def _init_sqlite(conn: sqlite3.Connection) -> None:
    """Create SQLite tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bucket_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            page_number INTEGER,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bucket_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            file_checksum TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            indexed_at TEXT DEFAULT (datetime('now')),
            UNIQUE(bucket_id, source_file)
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
            embedding float[1536] distance_metric=cosine
        )
    """)
    conn.commit()


def _get_sqlite_connection():
    """Create a connection to the SQLite database."""
    import sqlite_vec
    try:
        db_dir = os.path.dirname(DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _init_sqlite(conn)
        return conn
    except sqlite3.Error as e:
        logging.error(f"SQLite connection failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Postgres backend helpers
# ---------------------------------------------------------------------------

def _get_pg_connection():
    """Create a connection to the PostgreSQL database.
    Supports both TCP (host=IP) and Unix socket (host=/cloudsql/...) connections.
    """
    import psycopg2
    import psycopg2.extras
    try:
        connect_kwargs = dict(
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DATABASE,
        )
        if PG_HOST.startswith("/"):
            # Unix socket path (e.g., Cloud SQL: /cloudsql/project:region:instance)
            connect_kwargs["host"] = PG_HOST
        else:
            connect_kwargs["host"] = PG_HOST
            connect_kwargs["port"] = PG_PORT
        conn = psycopg2.connect(**connect_kwargs)
        conn.autocommit = False
        return conn
    except psycopg2.Error as e:
        logging.error(f"Postgres connection failed: {e}")
        raise


def _pg_fetchone(cur):
    """Fetch one row from a psycopg2 cursor as a dict."""
    row = cur.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in cur.description]
    return dict(zip(cols, row))


def _pg_fetchall(cur):
    """Fetch all rows from a psycopg2 cursor as dicts."""
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Unified connection dispatcher
# ---------------------------------------------------------------------------

def get_db_connection():
    """Get a database connection for the active backend."""
    if _is_postgres():
        return _get_pg_connection()
    return _get_sqlite_connection()


# Backward compat alias
init_db = _init_sqlite


# ---------------------------------------------------------------------------
# Shared helpers (backend-agnostic)
# ---------------------------------------------------------------------------

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
                        chunk_text_content = ' '.join(current_chunk)
                        chunks.append(Chunk(
                            text=chunk_text_content,
                            chunk_index=chunk_index,
                            page_number=page_num if len(pages) > 1 else None
                        ))
                        chunk_index += 1

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
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    chunk_text_content = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text_content,
                        chunk_index=chunk_index,
                        page_number=page_num if len(pages) > 1 else None
                    ))
                    chunk_index += 1

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
    batch_size = 100

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


# ---------------------------------------------------------------------------
# Public API — all functions dispatch internally by backend
# ---------------------------------------------------------------------------

def is_document_indexed(bucket_id: str, source_file: str, checksum: str) -> bool:
    """Check if a document with the given checksum is already indexed."""
    conn = get_db_connection()
    try:
        if _is_postgres():
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM indexed_documents WHERE bucket_id = %s AND source_file = %s AND file_checksum = %s",
                (bucket_id, source_file, checksum)
            )
            return cur.fetchone() is not None
        else:
            cur = conn.execute(
                "SELECT id FROM indexed_documents WHERE bucket_id = ? AND source_file = ? AND file_checksum = ?",
                (bucket_id, source_file, checksum)
            )
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

    if not force and is_document_indexed(bucket_id, source_file, file_checksum):
        logging.info(f"Document already indexed: {source_file}")
        return {"status": "skipped", "reason": "already_indexed", "checksum": file_checksum}

    chunks = chunk_text(text)
    if not chunks:
        return {"status": "error", "reason": "no_chunks_created"}

    logging.info(f"Created {len(chunks)} chunks from {source_file}")

    chunk_texts = [c.text for c in chunks]
    embeddings = get_embeddings(chunk_texts)

    logging.info(f"Generated embeddings for {len(embeddings)} chunks")

    conn = get_db_connection()
    try:
        if _is_postgres():
            _ingest_postgres(conn, bucket_id, source_file, file_checksum, chunks, embeddings, metadata, force)
        else:
            _ingest_sqlite(conn, bucket_id, source_file, file_checksum, chunks, embeddings, metadata, force)

        total_tokens = sum(estimate_tokens(c.text) for c in chunks)
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


def _ingest_sqlite(conn, bucket_id, source_file, file_checksum, chunks, embeddings, metadata, force):
    """SQLite-specific ingestion logic."""
    total_tokens = sum(estimate_tokens(c.text) for c in chunks)

    if force:
        cur = conn.execute(
            "SELECT id FROM embeddings WHERE bucket_id = ? AND source_file = ?",
            (bucket_id, source_file)
        )
        old_ids = [row['id'] for row in cur.fetchall()]
        for old_id in old_ids:
            conn.execute("DELETE FROM vec_embeddings WHERE rowid = ?", (old_id,))
        conn.execute(
            "DELETE FROM embeddings WHERE bucket_id = ? AND source_file = ?",
            (bucket_id, source_file)
        )
        conn.execute(
            "DELETE FROM indexed_documents WHERE bucket_id = ? AND source_file = ?",
            (bucket_id, source_file)
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk_metadata = metadata.copy() if metadata else {}
        embedding_blob = serialize_embedding(embedding)

        cur = conn.execute("""
            INSERT INTO embeddings
            (bucket_id, source_file, page_number, chunk_index, chunk_text, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            bucket_id, source_file, chunk.page_number, chunk.chunk_index,
            chunk.text, embedding_blob, json.dumps(chunk_metadata)
        ))

        row_id = cur.lastrowid
        conn.execute(
            "INSERT INTO vec_embeddings (rowid, embedding) VALUES (?, ?)",
            (row_id, embedding_blob)
        )

    conn.execute("""
        INSERT INTO indexed_documents
        (bucket_id, source_file, file_checksum, chunk_count, total_tokens, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (bucket_id, source_file)
        DO UPDATE SET
            file_checksum = excluded.file_checksum,
            chunk_count = excluded.chunk_count,
            total_tokens = excluded.total_tokens,
            metadata = excluded.metadata,
            indexed_at = datetime('now')
    """, (
        bucket_id, source_file, file_checksum, len(chunks),
        total_tokens, json.dumps(metadata or {})
    ))


def _ingest_postgres(conn, bucket_id, source_file, file_checksum, chunks, embeddings, metadata, force):
    """Postgres-specific ingestion logic."""
    total_tokens = sum(estimate_tokens(c.text) for c in chunks)
    cur = conn.cursor()

    if force:
        cur.execute(
            "DELETE FROM embeddings WHERE bucket_id = %s AND source_file = %s",
            (bucket_id, source_file)
        )
        cur.execute(
            "DELETE FROM indexed_documents WHERE bucket_id = %s AND source_file = %s",
            (bucket_id, source_file)
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk_metadata = metadata.copy() if metadata else {}
        # pgvector accepts a Python list directly as vector input
        cur.execute("""
            INSERT INTO embeddings
            (bucket_id, source_file, page_number, chunk_index, chunk_text, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
        """, (
            bucket_id, source_file, chunk.page_number, chunk.chunk_index,
            chunk.text, str(embedding), json.dumps(chunk_metadata)
        ))

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
        bucket_id, source_file, file_checksum, len(chunks),
        total_tokens, json.dumps(metadata or {})
    ))


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
    query_embedding = get_embeddings([query])[0]

    conn = get_db_connection()
    try:
        if _is_postgres():
            return _search_postgres(conn, query_embedding, bucket_id, limit, similarity_threshold)
        else:
            return _search_sqlite(conn, query_embedding, bucket_id, limit, similarity_threshold)
    finally:
        conn.close()


def _search_sqlite(conn, query_embedding, bucket_id, limit, similarity_threshold):
    """SQLite vector search via sqlite-vec virtual table."""
    query_blob = serialize_embedding(query_embedding)
    fetch_limit = limit * 5 if bucket_id else limit

    vec_rows = conn.execute(
        "SELECT rowid, distance FROM vec_embeddings WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (query_blob, fetch_limit)
    ).fetchall()

    results = []
    for vec_row in vec_rows:
        row_id = vec_row['rowid']
        distance = vec_row['distance']
        similarity = 1.0 - distance

        if similarity < similarity_threshold:
            continue

        row = conn.execute(
            "SELECT chunk_text, source_file, bucket_id, page_number, chunk_index, metadata "
            "FROM embeddings WHERE id = ?",
            (row_id,)
        ).fetchone()

        if row is None:
            continue

        if bucket_id and row['bucket_id'] != bucket_id:
            continue

        meta = row['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        results.append(SearchResult(
            chunk_text=row['chunk_text'],
            source_file=row['source_file'],
            bucket_id=row['bucket_id'],
            page_number=row['page_number'],
            chunk_index=row['chunk_index'],
            similarity=similarity,
            metadata=meta
        ))

        if len(results) >= limit:
            break

    return results


def _search_postgres(conn, query_embedding, bucket_id, limit, similarity_threshold):
    """Postgres vector search via pgvector <=> operator."""
    cur = conn.cursor()
    embedding_str = str(query_embedding)

    if bucket_id:
        cur.execute("""
            SELECT chunk_text, source_file, bucket_id, page_number, chunk_index, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM embeddings
            WHERE bucket_id = %s
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, bucket_id, embedding_str, similarity_threshold, embedding_str, limit))
    else:
        cur.execute("""
            SELECT chunk_text, source_file, bucket_id, page_number, chunk_index, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM embeddings
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, embedding_str, similarity_threshold, embedding_str, limit))

    results = []
    for row in _pg_fetchall(cur):
        meta = row['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        results.append(SearchResult(
            chunk_text=row['chunk_text'],
            source_file=row['source_file'],
            bucket_id=row['bucket_id'],
            page_number=row['page_number'],
            chunk_index=row['chunk_index'],
            similarity=float(row['similarity']),
            metadata=meta
        ))

    return results


def list_indexed_documents(bucket_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all indexed documents."""
    conn = get_db_connection()
    try:
        if _is_postgres():
            cur = conn.cursor()
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
            rows = _pg_fetchall(cur)
            for row in rows:
                if isinstance(row.get('indexed_at'), datetime):
                    row['indexed_at'] = row['indexed_at'].isoformat()
            return rows
        else:
            if bucket_id:
                cur = conn.execute("""
                    SELECT bucket_id, source_file, file_checksum, chunk_count,
                           total_tokens, indexed_at, metadata
                    FROM indexed_documents
                    WHERE bucket_id = ?
                    ORDER BY indexed_at DESC
                """, (bucket_id,))
            else:
                cur = conn.execute("""
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
    """Delete a document and its embeddings from the index."""
    conn = get_db_connection()
    try:
        if _is_postgres():
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM embeddings WHERE bucket_id = %s AND source_file = %s",
                (bucket_id, source_file)
            )
            cur.execute(
                "DELETE FROM indexed_documents WHERE bucket_id = %s AND source_file = %s",
                (bucket_id, source_file)
            )
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
        else:
            cur = conn.execute(
                "SELECT id FROM embeddings WHERE bucket_id = ? AND source_file = ?",
                (bucket_id, source_file)
            )
            embedding_ids = [row['id'] for row in cur.fetchall()]
            for eid in embedding_ids:
                conn.execute("DELETE FROM vec_embeddings WHERE rowid = ?", (eid,))
            conn.execute(
                "DELETE FROM embeddings WHERE bucket_id = ? AND source_file = ?",
                (bucket_id, source_file)
            )
            cur = conn.execute(
                "DELETE FROM indexed_documents WHERE bucket_id = ? AND source_file = ?",
                (bucket_id, source_file)
            )
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    finally:
        conn.close()


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the RAG index."""
    conn = get_db_connection()
    try:
        if _is_postgres():
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM indexed_documents")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM embeddings")
            chunk_count = cur.fetchone()[0]
            cur.execute("SELECT COALESCE(SUM(total_tokens), 0) FROM indexed_documents")
            total_tokens = cur.fetchone()[0]
            cur.execute("SELECT bucket_id, COUNT(*) as count FROM indexed_documents GROUP BY bucket_id")
            by_bucket = {row[0]: row[1] for row in cur.fetchall()}
        else:
            doc_count = conn.execute("SELECT COUNT(*) as doc_count FROM indexed_documents").fetchone()['doc_count']
            chunk_count = conn.execute("SELECT COUNT(*) as chunk_count FROM embeddings").fetchone()['chunk_count']
            total_tokens = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) as total_tokens FROM indexed_documents"
            ).fetchone()['total_tokens']
            rows = conn.execute("""
                SELECT bucket_id, COUNT(*) as count
                FROM indexed_documents
                GROUP BY bucket_id
            """).fetchall()
            by_bucket = {row['bucket_id']: row['count'] for row in rows}

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "total_tokens": total_tokens,
            "documents_by_bucket": by_bucket,
            "backend": RAG_BACKEND
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="RAG operations for PDFScribe")
    parser.add_argument("--backend", choices=["sqlite", "postgres"],
                        help="Override RAG_BACKEND env var")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest file command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document file")
    ingest_parser.add_argument("file", help="Text file to ingest")
    ingest_parser.add_argument("--bucket", required=True, help="Context bucket ID")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-index")

    # Ingest text from stdin
    ingest_text_parser = subparsers.add_parser("ingest-text", help="Ingest text from stdin")
    ingest_text_parser.add_argument("--bucket", required=True, help="Context bucket ID")
    ingest_text_parser.add_argument("--source", required=True, help="Source identifier")
    ingest_text_parser.add_argument("--force", action="store_true", help="Force re-index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--bucket", help="Filter by context bucket")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")
    search_parser.add_argument("--threshold", type=float, default=0.5, help="Min similarity")

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

    # Override backend if specified
    if args.backend:
        RAG_BACKEND = args.backend

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

    elif args.command == "ingest-text":
        text = sys.stdin.read()
        if not text.strip():
            print("Error: no text provided on stdin", file=sys.stderr)
            sys.exit(1)
        result = ingest_document(
            text=text,
            bucket_id=args.bucket,
            source_file=args.source,
            force=args.force
        )
        print(json.dumps(result, indent=2))

    elif args.command == "search":
        results = search_documents(
            query=args.query,
            bucket_id=args.bucket,
            limit=args.limit,
            similarity_threshold=args.threshold
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
