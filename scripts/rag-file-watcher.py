#!/usr/bin/env python3
"""Watch context bucket files/ directories and auto-ingest on change.

Runs as a daemon (launchd on macOS). When a .md, .txt, or .html file is
created or modified in any context-buckets/<bucket>/files/ directory, it
is automatically ingested into the cloud RAG database.

Requires:
    pip install watchdog
    Environment: RAG_BACKEND=postgres + DB connection vars + OPENAI_API_KEY
"""

import os
import sys
import time
import logging
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add parent for rag imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rag-watcher")

BUCKETS_ROOT = Path.home() / "Workspaces" / "AgentArchitect" / "context-buckets"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".html"}
SKIP_FILES = {"home.html", "index.html", ".DS_Store"}
DEBOUNCE_SECONDS = 5


class BucketIngestHandler(FileSystemEventHandler):
    """Handles file create/modify events in context bucket files/ dirs."""

    def __init__(self):
        self._pending = {}  # path -> timestamp (debounce)

    def _should_process(self, path: str) -> bool:
        p = Path(path)
        if p.name in SKIP_FILES:
            return False
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        if "/files/" not in str(p):
            return False
        return True

    def _get_bucket_id(self, path: str) -> str | None:
        """Extract bucket_id from path like .../context-buckets/<bucket-id>/files/..."""
        parts = Path(path).parts
        try:
            idx = parts.index("context-buckets")
            return parts[idx + 1]
        except (ValueError, IndexError):
            return None

    def _ingest_file(self, path: str):
        from rag import ingest_document

        p = Path(path)
        bucket_id = self._get_bucket_id(path)
        if not bucket_id:
            log.warning(f"Could not determine bucket_id for {path}")
            return

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 50:
                log.info(f"Skipping {p.name} (too short)")
                return

            result = ingest_document(
                text=text,
                bucket_id=bucket_id,
                source_file=p.name,
                force=True,
            )
            log.info(
                f"Ingested {p.name} into {bucket_id}: "
                f"{result.get('chunk_count', 0)} chunks, "
                f"{result.get('total_tokens', 0)} tokens"
            )
        except Exception:
            log.exception(f"Failed to ingest {path}")

    def on_created(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return
        self._pending[event.src_path] = time.time()

    def on_modified(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return
        self._pending[event.src_path] = time.time()

    def process_pending(self):
        """Process files that have been stable for DEBOUNCE_SECONDS."""
        now = time.time()
        ready = [
            p for p, t in self._pending.items()
            if now - t >= DEBOUNCE_SECONDS
        ]
        for path in ready:
            del self._pending[path]
            if Path(path).exists():
                log.info(f"Auto-ingesting: {path}")
                self._ingest_file(path)


def main():
    os.environ.setdefault("RAG_BACKEND", "postgres")

    if not BUCKETS_ROOT.exists():
        log.error(f"Buckets root not found: {BUCKETS_ROOT}")
        sys.exit(1)

    # Discover all bucket files/ directories
    watch_dirs = []
    for bucket_dir in sorted(BUCKETS_ROOT.iterdir()):
        if bucket_dir.name.startswith("_"):
            continue
        files_dir = bucket_dir / "files"
        if files_dir.is_dir():
            watch_dirs.append(str(files_dir))

    if not watch_dirs:
        log.error("No bucket files/ directories found")
        sys.exit(1)

    handler = BucketIngestHandler()
    observer = Observer()

    for d in watch_dirs:
        observer.schedule(handler, d, recursive=True)
        log.info(f"Watching: {d}")

    log.info(f"RAG File Watcher started — monitoring {len(watch_dirs)} buckets")
    observer.start()

    try:
        while True:
            time.sleep(1)
            handler.process_pending()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        observer.stop()
    observer.join()
    log.info("Stopped.")


if __name__ == "__main__":
    main()
