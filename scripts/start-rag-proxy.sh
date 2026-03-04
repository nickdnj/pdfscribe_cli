#!/bin/bash
# Start Cloud SQL Auth Proxy for local RAG access
# The proxy connects to the pdfscribe-db Cloud SQL instance
# and exposes it on localhost:5433

echo "Starting Cloud SQL Auth Proxy for RAG database..."
echo "  Instance: pdfscribe-prod:us-central1:pdfscribe-db"
echo "  Local port: 5433"
echo ""

exec cloud-sql-proxy pdfscribe-prod:us-central1:pdfscribe-db --port=5433
