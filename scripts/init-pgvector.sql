-- RAG Cloud Database Schema (PostgreSQL + pgvector)
-- Run against the 'rag' database on pdfscribe-db Cloud SQL instance

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings table (stores chunks + vectors)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    bucket_id TEXT NOT NULL,
    source_file TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexed documents tracking table
CREATE TABLE IF NOT EXISTS indexed_documents (
    id SERIAL PRIMARY KEY,
    bucket_id TEXT NOT NULL,
    source_file TEXT NOT NULL,
    file_checksum TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(bucket_id, source_file)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_embeddings_bucket ON embeddings (bucket_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings (bucket_id, source_file);
CREATE INDEX IF NOT EXISTS idx_indexed_docs_bucket ON indexed_documents (bucket_id);

-- HNSW vector index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Grant permissions to rag user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag;
