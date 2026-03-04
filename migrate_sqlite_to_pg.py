#!/usr/bin/env python3
"""
Migrate RAG data from SQLite (sqlite-vec) to PostgreSQL (pgvector).
One-time migration script.

Usage:
    python migrate_sqlite_to_pg.py [--verify-only] [--batch-size 100]

Requires:
    - Cloud SQL Auth Proxy running on port 5433
    - RAG_DB_PASSWORD env var set
    - SQLite db at ~/.wharfside/rag.db
"""

import os
import sys
import json
import struct
import sqlite3
import logging
from pathlib import Path

import sqlite_vec
import psycopg2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# SQLite source
SQLITE_PATH = os.getenv("RAG_DB_PATH", os.path.expanduser("~/.wharfside/rag.db"))

# Postgres target
PG_HOST = os.getenv("RAG_DB_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("RAG_DB_PORT", "5433"))
PG_USER = os.getenv("RAG_DB_USER", "rag")
PG_PASSWORD = os.getenv("RAG_DB_PASSWORD", "")
PG_DATABASE = os.getenv("RAG_DB_NAME", "rag")


def deserialize_embedding(blob: bytes) -> list:
    """Deserialize sqlite-vec embedding bytes to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DATABASE
    )


def migrate(batch_size=100):
    """Migrate all data from SQLite to Postgres."""
    sqlite_conn = get_sqlite_conn()
    pg_conn = get_pg_conn()
    pg_cur = pg_conn.cursor()

    # --- Migrate indexed_documents ---
    log.info("Migrating indexed_documents...")
    rows = sqlite_conn.execute("SELECT * FROM indexed_documents").fetchall()
    doc_count = 0
    for row in rows:
        meta = row['metadata']
        if isinstance(meta, str):
            meta = meta if meta else '{}'

        pg_cur.execute("""
            INSERT INTO indexed_documents
            (bucket_id, source_file, file_checksum, chunk_count, total_tokens, metadata, indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bucket_id, source_file) DO NOTHING
        """, (
            row['bucket_id'], row['source_file'], row['file_checksum'],
            row['chunk_count'], row['total_tokens'], meta, row['indexed_at']
        ))
        doc_count += 1
    pg_conn.commit()
    log.info(f"  Migrated {doc_count} documents")

    # --- Migrate embeddings ---
    log.info("Migrating embeddings...")
    total_rows = sqlite_conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    log.info(f"  Total chunks to migrate: {total_rows}")

    offset = 0
    migrated = 0
    while offset < total_rows:
        rows = sqlite_conn.execute(
            "SELECT * FROM embeddings ORDER BY id LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()

        if not rows:
            break

        for row in rows:
            embedding_list = deserialize_embedding(row['embedding'])
            meta = row['metadata']
            if isinstance(meta, str):
                meta = meta if meta else '{}'

            pg_cur.execute("""
                INSERT INTO embeddings
                (bucket_id, source_file, page_number, chunk_index, chunk_text, embedding, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
            """, (
                row['bucket_id'], row['source_file'], row['page_number'],
                row['chunk_index'], row['chunk_text'], str(embedding_list),
                meta, row['created_at']
            ))
            migrated += 1

        pg_conn.commit()
        offset += batch_size
        log.info(f"  Migrated {migrated}/{total_rows} chunks")

    log.info(f"Migration complete: {doc_count} documents, {migrated} chunks")

    sqlite_conn.close()
    pg_conn.close()


def verify():
    """Verify row counts match between SQLite and Postgres."""
    sqlite_conn = get_sqlite_conn()
    pg_conn = get_pg_conn()
    pg_cur = pg_conn.cursor()

    log.info("Verifying migration...")

    # Document counts
    sqlite_docs = sqlite_conn.execute("SELECT COUNT(*) FROM indexed_documents").fetchone()[0]
    pg_cur.execute("SELECT COUNT(*) FROM indexed_documents")
    pg_docs = pg_cur.fetchone()[0]
    match_docs = "OK" if sqlite_docs == pg_docs else "MISMATCH"
    log.info(f"  indexed_documents: SQLite={sqlite_docs}, Postgres={pg_docs} [{match_docs}]")

    # Chunk counts
    sqlite_chunks = sqlite_conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    pg_cur.execute("SELECT COUNT(*) FROM embeddings")
    pg_chunks = pg_cur.fetchone()[0]
    match_chunks = "OK" if sqlite_chunks == pg_chunks else "MISMATCH"
    log.info(f"  embeddings: SQLite={sqlite_chunks}, Postgres={pg_chunks} [{match_chunks}]")

    # Per-bucket counts
    sqlite_buckets = sqlite_conn.execute(
        "SELECT bucket_id, COUNT(*) FROM indexed_documents GROUP BY bucket_id"
    ).fetchall()
    for row in sqlite_buckets:
        pg_cur.execute(
            "SELECT COUNT(*) FROM indexed_documents WHERE bucket_id = %s",
            (row[0],)
        )
        pg_count = pg_cur.fetchone()[0]
        match = "OK" if row[1] == pg_count else "MISMATCH"
        log.info(f"  bucket '{row[0]}': SQLite={row[1]}, Postgres={pg_count} [{match}]")

    sqlite_conn.close()
    pg_conn.close()

    return sqlite_docs == pg_docs and sqlite_chunks == pg_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate RAG data from SQLite to Postgres")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't migrate")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for inserts")
    args = parser.parse_args()

    if not PG_PASSWORD:
        log.error("RAG_DB_PASSWORD env var is required")
        sys.exit(1)

    if args.verify_only:
        ok = verify()
        sys.exit(0 if ok else 1)
    else:
        migrate(batch_size=args.batch_size)
        verify()
