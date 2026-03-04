#!/usr/bin/env python3
"""
Universal context bucket ingestion for the RAG shared brain.
Auto-discovers buckets from AgentArchitect/context-buckets/ and ingests
HTML, Markdown, and text files.

Usage:
    python ingest_all_buckets.py                  # Ingest all buckets
    python ingest_all_buckets.py --bucket research-cache  # Single bucket
    python ingest_all_buckets.py --dry-run         # Preview only
    python ingest_all_buckets.py --force            # Re-index everything
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from html.parser import HTMLParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import ingest_document, get_index_stats, compute_file_checksum

BUCKETS_DIR = Path(__file__).parent.parent / "AgentArchitect" / "context-buckets"

# Files to skip globally
SKIP_FILES = {"home.html", "index.html", ".DS_Store"}

# File extensions to ingest
INGESTABLE_EXTENSIONS = {".html", ".md", ".txt"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, preserving paragraph breaks."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {"script", "style", "head"}
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip_depth += 1
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self.result.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.result.append(data)

    def get_text(self):
        text = "".join(self.result)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()


def extract_html_text(filepath):
    """Extract text content from an HTML file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        html = f.read()
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def discover_buckets():
    """Discover all context buckets with bucket.json."""
    buckets = []
    if not BUCKETS_DIR.exists():
        log.error(f"Buckets directory not found: {BUCKETS_DIR}")
        return buckets

    for entry in sorted(BUCKETS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith('_'):
            continue
        bucket_json = entry / "bucket.json"
        if bucket_json.exists():
            with open(bucket_json) as f:
                config = json.load(f)
            buckets.append({
                "id": config.get("id", entry.name),
                "name": config.get("name", entry.name),
                "path": entry,
                "files_dir": entry / "files",
            })
    return buckets


def collect_files(files_dir):
    """Recursively collect ingestable files from a directory."""
    files = []
    if not files_dir.exists():
        return files

    for filepath in sorted(files_dir.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.name in SKIP_FILES:
            continue
        if filepath.suffix.lower() in INGESTABLE_EXTENSIONS:
            files.append(filepath)
    return files


def extract_text(filepath):
    """Extract text from a file based on its extension."""
    ext = filepath.suffix.lower()
    if ext == '.html':
        return extract_html_text(filepath)
    elif ext in ('.md', '.txt'):
        return filepath.read_text(encoding='utf-8', errors='replace')
    return None


def friendly_name(filepath, files_dir):
    """Create a friendly source name relative to the files directory."""
    try:
        return str(filepath.relative_to(files_dir))
    except ValueError:
        return filepath.name


def ingest_bucket(bucket, force=False, dry_run=False):
    """Ingest all files from a single bucket."""
    bucket_id = bucket["id"]
    files_dir = bucket["files_dir"]
    files = collect_files(files_dir)

    if not files:
        log.info(f"  No ingestable files found")
        return {"success": 0, "skipped": 0, "errors": 0}

    log.info(f"  Found {len(files)} files")

    success = 0
    skipped = 0
    errors = 0

    for filepath in files:
        source_name = friendly_name(filepath, files_dir)

        if dry_run:
            log.info(f"    [DRY] {source_name} ({filepath.suffix})")
            continue

        try:
            text = extract_text(filepath)
            if not text or len(text.strip()) < 50:
                log.info(f"    [SKIP] {source_name} (too short)")
                skipped += 1
                continue

            checksum = compute_file_checksum(str(filepath))
            result = ingest_document(
                text=text,
                bucket_id=bucket_id,
                source_file=source_name,
                file_checksum=checksum,
                metadata={"file_type": filepath.suffix.lstrip('.')},
                force=force
            )

            status = result.get("status", "unknown")
            if status == "success":
                chunks = result.get("chunk_count", 0)
                tokens = result.get("total_tokens", 0)
                log.info(f"    [OK] {source_name} ({chunks} chunks, {tokens} tokens)")
                success += 1
            elif status == "skipped":
                log.info(f"    [SKIP] {source_name} (unchanged)")
                skipped += 1
            else:
                log.info(f"    [ERROR] {source_name}: {result}")
                errors += 1

        except Exception as e:
            log.error(f"    [ERROR] {source_name}: {e}")
            errors += 1

    return {"success": success, "skipped": skipped, "errors": errors}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest all context buckets into RAG")
    parser.add_argument("--bucket", help="Ingest only this bucket ID")
    parser.add_argument("--force", action="store_true", help="Force re-index all files")
    parser.add_argument("--dry-run", action="store_true", help="Preview files without ingesting")
    args = parser.parse_args()

    log.info("Universal Context Bucket Ingestion")
    log.info("=" * 55)

    buckets = discover_buckets()
    if args.bucket:
        buckets = [b for b in buckets if b["id"] == args.bucket]
        if not buckets:
            log.error(f"Bucket not found: {args.bucket}")
            sys.exit(1)

    log.info(f"Discovered {len(buckets)} bucket(s)")
    if args.dry_run:
        log.info("[DRY RUN MODE]")
    log.info("")

    totals = {"success": 0, "skipped": 0, "errors": 0}

    for bucket in buckets:
        log.info(f"[{bucket['id']}] {bucket['name']}")
        result = ingest_bucket(bucket, force=args.force, dry_run=args.dry_run)
        for key in totals:
            totals[key] += result[key]
        log.info("")

    log.info("=" * 55)
    log.info(f"Total: {totals['success']} ingested, {totals['skipped']} skipped, {totals['errors']} errors")

    if not args.dry_run:
        log.info("")
        stats = get_index_stats()
        log.info(f"RAG Index Stats ({stats.get('backend', 'unknown')} backend):")
        log.info(f"  Documents: {stats['total_documents']}")
        log.info(f"  Chunks:    {stats['total_chunks']}")
        log.info(f"  Tokens:    {stats['total_tokens']}")
        log.info(f"  Buckets:   {json.dumps(stats['documents_by_bucket'], indent=4)}")


if __name__ == "__main__":
    main()
