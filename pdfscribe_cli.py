#!/usr/bin/env python3
"""
PDFScribe CLI - PDF transcription using vision AI capabilities
Converts scanned PDFs to text using Anthropic's Claude API or OpenAI's GPT-4 Vision

Features:
- Multi-page PDF transcription
- Caching of transcriptions next to source documents
- Checksum-based cache validation
- Google Drive integration (download source, upload transcription)
- Support for multiple AI providers (Anthropic, OpenAI)
"""
import argparse
import os
import io
import base64
import logging
import hashlib
import re
import json
from datetime import datetime, timezone
import requests
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env (if present)
load_dotenv()

# AI Provider configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic").lower()  # 'anthropic' or 'openai'
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys based on provider
if AI_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    logging.warning("ANTHROPIC_API_KEY is not set. Please set it in your environment or .env file.")
elif AI_PROVIDER == "openai" and not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")

# Google Drive credentials path
GDRIVE_CREDENTIALS_PATH = os.path.expanduser("~/.config/mcp-gdrive/.gdrive-server-credentials.json")
GDRIVE_OAUTH_PATH = os.path.expanduser("~/.config/mcp-gdrive/gcp-oauth.keys.json")

# Global parameters used in the prompt and API call
PROMPT_TEXT = (
    "You are a skilled transcriber specializing in extracting text from poor-quality scanned document pages. "
    "Your task is to accurately transcribe all visible text and handle any handwritten annotations such as dates, "
    "checkmarks, or initials with special attention:\n"
    "Annotations: Enclose handwritten annotations in curly brackets. For example, for handwritten initials, transcribe them as {initials}.\n"
    "Uncertain Text: If you encounter any text that you are unsure about, highlight it in red to indicate uncertainty.\n"
    "Formatting:\n"
    "Tables: Use simple HTML tags to represent tables.\n"
    "Bold Text: Use HTML tags to bold any text that appears bold in the original document.\n"
    "Notes: Focus solely on transcription without adding any notes or commentary."
)

# Provider-specific parameters
PARAMETERS = {
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "openai": {
        "model": "gpt-4.1-mini",  # Fast and cost-effective vision model
        "max_tokens": 4096,
        "temperature": 0.7
    }
}

def get_current_provider():
    """Get the current AI provider setting."""
    return AI_PROVIDER

def get_current_model():
    """Get the model for the current provider."""
    return PARAMETERS[AI_PROVIDER]["model"]

# ============================================================================
# Google Drive Functions
# ============================================================================

def get_gdrive_service():
    """Initialize and return Google Drive service."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        logging.error("Google API libraries not installed. Run: pip install google-api-python-client google-auth")
        return None

    if not os.path.exists(GDRIVE_CREDENTIALS_PATH) or not os.path.exists(GDRIVE_OAUTH_PATH):
        logging.error(f"Google Drive credentials not found at {GDRIVE_CREDENTIALS_PATH}")
        return None

    with open(GDRIVE_OAUTH_PATH) as f:
        oauth_keys = json.load(f)['installed']

    with open(GDRIVE_CREDENTIALS_PATH) as f:
        tokens = json.load(f)

    creds = Credentials(
        token=tokens['access_token'],
        refresh_token=tokens['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token',
        client_id=oauth_keys['client_id'],
        client_secret=oauth_keys['client_secret']
    )

    return build('drive', 'v3', credentials=creds)


def get_file_parent_folder(service, file_id):
    """Get the parent folder ID of a file in Google Drive."""
    try:
        file_metadata = service.files().get(fileId=file_id, fields='parents').execute()
        parents = file_metadata.get('parents', [])
        return parents[0] if parents else None
    except Exception as e:
        logging.error(f"Failed to get parent folder: {e}")
        return None


def download_from_gdrive(service, file_id, local_path):
    """Download a file from Google Drive."""
    from googleapiclient.http import MediaIoBaseDownload

    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logging.debug(f"Download progress: {int(status.progress() * 100)}%")

        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())

        logging.info(f"Downloaded from Google Drive: {local_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download from Google Drive: {e}")
        return False


def upload_to_gdrive(service, local_path, folder_id, filename=None):
    """
    Upload a file to Google Drive in the specified folder.
    Returns the file ID of the uploaded file, or None on failure.
    """
    from googleapiclient.http import MediaFileUpload

    if filename is None:
        filename = os.path.basename(local_path)

    # Check if file already exists in folder
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    try:
        results = service.files().list(q=query, fields='files(id, name)').execute()
        existing_files = results.get('files', [])
    except Exception as e:
        logging.error(f"Failed to check for existing file: {e}")
        existing_files = []

    file_metadata = {
        'name': filename,
        'mimeType': 'text/markdown'
    }

    media = MediaFileUpload(local_path, mimetype='text/markdown', resumable=True)

    try:
        if existing_files:
            # Update existing file
            file_id = existing_files[0]['id']
            updated_file = service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            logging.info(f"Updated existing file in Google Drive: {filename}")
            return updated_file.get('id')
        else:
            # Create new file
            file_metadata['parents'] = [folder_id]
            new_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            logging.info(f"Uploaded new file to Google Drive: {filename}")
            return new_file.get('id')
    except Exception as e:
        logging.error(f"Failed to upload to Google Drive: {e}")
        return None


# ============================================================================
# Caching Functions
# ============================================================================

def compute_checksum(file_path):
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_cache_path(pdf_path):
    """
    Get the cache file path for a given PDF.
    Cache is stored next to the source document as {filename}-transcribed.md
    """
    directory = os.path.dirname(pdf_path) or "."
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(directory, f"{basename}-transcribed.md")


def parse_cache_metadata(cache_path):
    """
    Parse metadata from the cache file header.
    Returns dict with metadata or None if invalid/missing.
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            content = f.read(2000)  # Read enough for metadata header

        # Look for metadata block
        match = re.search(r'<!--\s*PDFScribe Cache\s*(.*?)-->', content, re.DOTALL)
        if not match:
            return None

        metadata_block = match.group(1)
        metadata = {}

        # Parse key: value pairs
        for line in metadata_block.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip().lower()] = value.strip()

        return metadata
    except Exception as e:
        logging.debug(f"Error parsing cache metadata: {e}")
        return None


def is_cache_valid(pdf_path, cache_path):
    """
    Check if cache is valid by comparing checksums.
    Returns True if cache exists and checksum matches.
    """
    metadata = parse_cache_metadata(cache_path)
    if not metadata:
        return False

    cached_checksum = metadata.get('checksum')
    if not cached_checksum:
        return False

    current_checksum = compute_checksum(pdf_path)
    return cached_checksum == current_checksum


def read_cached_transcription(cache_path):
    """
    Read the cached transcription, stripping the metadata header.
    Returns the transcription content.
    """
    with open(cache_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove metadata block
    content = re.sub(r'<!--\s*PDFScribe Cache\s*.*?-->\s*', '', content, flags=re.DOTALL)
    return content.strip()


def write_cached_transcription(cache_path, content, pdf_path, original_name, page_count, model, checksum, gdrive_id=None):
    """
    Write transcription to cache file with metadata header.
    """
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    gdrive_line = f"\nGDrive-Source: {gdrive_id}" if gdrive_id else ""

    metadata_header = f"""<!--
PDFScribe Cache
Source: {pdf_path}
Original: {original_name}
Transcribed: {timestamp}
Model: {model}
Pages: {page_count}
Checksum: {checksum}{gdrive_line}
-->

"""

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(metadata_header + content)

    logging.info(f"Cache written to: {cache_path}")


# ============================================================================
# PDF Processing Functions
# ============================================================================

def pdf_to_images(pdf_bytes):
    """Convert PDF bytes to a list of PIL image objects (one per page)."""
    logging.info("Converting PDF to images...")
    return convert_from_bytes(pdf_bytes)


def encode_image(image):
    """Encode a PIL image to a base64 PNG string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    logging.debug("Image encoded to base64.")
    return encoded_image


def image_to_text_anthropic(base64_image, page_number, backstory):
    """
    Send the encoded image to Claude via the Anthropic API.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    prompt_text = (backstory + "\n\n" if backstory else "") + PROMPT_TEXT
    params = PARAMETERS["anthropic"]

    payload = {
        "model": params['model'],
        "max_tokens": params['max_tokens'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code == 200:
        result = response.json()
        content = result.get('content', [])
        text_parts = [block['text'] for block in content if block.get('type') == 'text']
        return '\n'.join(text_parts), None
    else:
        return None, response


def image_to_text_openai(base64_image, page_number, backstory):
    """
    Send the encoded image to OpenAI's GPT-4 Vision API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    prompt_text = (backstory + "\n\n" if backstory else "") + PROMPT_TEXT
    params = PARAMETERS["openai"]

    payload = {
        "model": params['model'],
        "max_tokens": params['max_tokens'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code == 200:
        result = response.json()
        choices = result.get('choices', [])
        if choices:
            return choices[0]['message']['content'], None
        return "", None
    else:
        return None, response


def image_to_text(base64_image, page_number, backstory):
    """
    Send the encoded image along with a prompt (and optional backstory)
    to the configured AI provider and return the transcribed text.
    """
    # Select the appropriate provider function
    if AI_PROVIDER == "openai":
        provider_func = image_to_text_openai
    else:
        provider_func = image_to_text_anthropic

    # Try up to 5 times to process the image
    for attempt in range(5):
        logging.info(f"Processing page {page_number}: Attempt {attempt + 1} ({AI_PROVIDER})")
        try:
            text, error_response = provider_func(base64_image, page_number, backstory)

            if text is not None:
                logging.info(f"Page {page_number} processed successfully.")
                return text
            else:
                error_info = error_response.json() if error_response.text else {"error": error_response.status_code}
                logging.error(f"Attempt {attempt + 1} failed for page {page_number}: {error_info}")

                # If rate limited, wait before retry
                if error_response.status_code == 429:
                    import time
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logging.info(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        except requests.exceptions.Timeout:
            logging.error(f"Attempt {attempt + 1} timed out for page {page_number}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for page {page_number}: {e}")

    logging.error(f"Page {page_number} failed after 5 attempts.")
    return f"[Error processing page {page_number}]"


def transcribe_pdf(pdf_path, backstory="", use_cache=True, force=False, gdrive_id=None):
    """
    Transcribe a PDF file, using cache if available and valid.

    Args:
        pdf_path: Path to the PDF file
        backstory: Optional context for transcription
        use_cache: Whether to use/create cache (default: True)
        force: Force re-transcription even if cache exists (default: False)
        gdrive_id: Google Drive file ID of source (for metadata)

    Returns:
        Tuple of (transcription_text, cache_path, from_cache)
    """
    cache_path = get_cache_path(pdf_path)

    # Check cache (unless disabled or forced)
    if use_cache and not force and os.path.exists(cache_path):
        if is_cache_valid(pdf_path, cache_path):
            logging.info(f"Using cached transcription: {cache_path}")
            content = read_cached_transcription(cache_path)
            return content, cache_path, True
        else:
            logging.info("Cache exists but checksum mismatch - re-transcribing")

    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Convert to images
    try:
        images = pdf_to_images(pdf_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")

    logging.info(f"Processing {len(images)} page(s) with {get_current_model()} ({AI_PROVIDER})...")

    # Process each page
    all_text = []
    for i, image in enumerate(tqdm(images, desc="Processing pages", unit="page")):
        page_number = i + 1
        base64_image = encode_image(image)
        text = image_to_text(base64_image, page_number, backstory)
        all_text.append(text)

    # Combine all pages
    transcription = "\n\n--- Page Break ---\n\n".join(all_text)

    # Write cache if enabled
    if use_cache:
        checksum = compute_checksum(pdf_path)
        original_name = os.path.basename(pdf_path)
        write_cached_transcription(
            cache_path,
            transcription,
            pdf_path,
            original_name,
            len(images),
            get_current_model(),
            checksum,
            gdrive_id
        )

    return transcription, cache_path, False


def main():
    global AI_PROVIDER  # Allow overriding via CLI

    parser = argparse.ArgumentParser(
        description="Process a scanned PDF file page by page with AI vision and save the transcription results."
    )
    parser.add_argument("pdf_file", nargs='?', help="Path to the scanned PDF file (or use --gdrive)")
    parser.add_argument(
        "-o", "--output", help="Output file to save the transcription results (default: auto-cached next to source)",
        default=None
    )
    parser.add_argument(
        "-b", "--backstory", help="Optional backstory text to prepend to the prompt", default=""
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging for debugging"
    )
    parser.add_argument(
        "-m", "--model", help="Model to use (default depends on provider)",
        default=None
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"],
        help="AI provider to use (default: anthropic, or AI_PROVIDER env var)",
        default=None
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching - don't read or write cache files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-transcription even if valid cache exists"
    )
    parser.add_argument(
        "--gdrive", metavar="FILE_ID",
        help="Google Drive file ID to download and transcribe"
    )
    parser.add_argument(
        "--gdrive-upload", action="store_true",
        help="Upload transcription to Google Drive (next to source PDF)"
    )
    parser.add_argument(
        "--gdrive-folder", metavar="FOLDER_ID",
        help="Google Drive folder ID to upload transcription to (overrides auto-detect)"
    )
    parser.add_argument(
        "--work-dir", default="/tmp/pdfscribe",
        help="Working directory for downloaded files (default: /tmp/pdfscribe)"
    )

    # RAG (Retrieval-Augmented Generation) options
    parser.add_argument(
        "--ingest", action="store_true",
        help="Ingest transcription into RAG system after transcribing"
    )
    parser.add_argument(
        "--bucket", metavar="BUCKET_ID",
        help="Context bucket ID for RAG ingestion (required with --ingest)"
    )
    parser.add_argument(
        "--rag-force", action="store_true",
        help="Force re-ingest even if document already indexed"
    )

    args = parser.parse_args()

    # Override provider if specified via CLI
    if args.provider:
        AI_PROVIDER = args.provider

    # Override model if specified
    if args.model:
        PARAMETERS[AI_PROVIDER]['model'] = args.model

    # Configure logging level
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Validate API key for the selected provider
    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logging.error("OPENAI_API_KEY environment variable is required when using OpenAI provider.")
            logging.error("Get your API key at: https://platform.openai.com/api-keys")
            logging.error("Then set it: export OPENAI_API_KEY='your-key'")
            return
    else:  # anthropic
        if not ANTHROPIC_API_KEY:
            logging.error("ANTHROPIC_API_KEY environment variable is required.")
            logging.error("Get your API key at: https://console.anthropic.com/")
            logging.error("Then set it: export ANTHROPIC_API_KEY='your-key'")
            return

    logging.info(f"Using AI provider: {AI_PROVIDER}")

    # Handle Google Drive source
    gdrive_service = None
    gdrive_source_id = None
    gdrive_folder_id = args.gdrive_folder

    if args.gdrive:
        gdrive_service = get_gdrive_service()
        if not gdrive_service:
            logging.error("Failed to initialize Google Drive service")
            return

        gdrive_source_id = args.gdrive

        # Get file metadata
        try:
            file_metadata = gdrive_service.files().get(
                fileId=gdrive_source_id,
                fields='name, parents'
            ).execute()
            filename = file_metadata.get('name', f'{gdrive_source_id}.pdf')
            if not gdrive_folder_id:
                parents = file_metadata.get('parents', [])
                gdrive_folder_id = parents[0] if parents else None
        except Exception as e:
            logging.error(f"Failed to get file metadata: {e}")
            return

        # Create work directory
        os.makedirs(args.work_dir, exist_ok=True)
        pdf_path = os.path.join(args.work_dir, filename)

        # Download PDF
        logging.info(f"Downloading {filename} from Google Drive...")
        if not download_from_gdrive(gdrive_service, gdrive_source_id, pdf_path):
            return

        args.pdf_file = pdf_path

    if not args.pdf_file:
        parser.error("Either pdf_file or --gdrive FILE_ID is required")

    if not os.path.isfile(args.pdf_file):
        logging.error(f"PDF file '{args.pdf_file}' does not exist.")
        return

    # Transcribe with caching
    use_cache = not args.no_cache
    try:
        transcription, cache_path, from_cache = transcribe_pdf(
            args.pdf_file,
            backstory=args.backstory,
            use_cache=use_cache,
            force=args.force,
            gdrive_id=gdrive_source_id
        )
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return

    # If output differs from cache, write separately
    if args.output and args.output != cache_path:
        try:
            with open(args.output, "w", encoding="utf-8") as out_file:
                out_file.write(transcription)
            logging.info(f"Transcription saved to '{args.output}'.")
        except Exception as e:
            logging.error(f"Failed to write output to file: {e}")
            return

    if from_cache:
        print(f"✓ Transcription retrieved from cache: {cache_path}")
    else:
        print(f"✓ Transcription complete. Cached at: {cache_path}")

    # Upload to Google Drive if requested
    if args.gdrive_upload or (args.gdrive and gdrive_folder_id):
        if not gdrive_service:
            gdrive_service = get_gdrive_service()

        if gdrive_service and gdrive_folder_id:
            md_filename = os.path.basename(cache_path)
            uploaded_id = upload_to_gdrive(gdrive_service, cache_path, gdrive_folder_id, md_filename)
            if uploaded_id:
                print(f"✓ Uploaded to Google Drive: {md_filename} (ID: {uploaded_id})")
            else:
                print("✗ Failed to upload to Google Drive")
        elif not gdrive_folder_id:
            logging.error("No Google Drive folder specified. Use --gdrive-folder FOLDER_ID")

    # RAG ingestion if requested
    if args.ingest:
        if not args.bucket:
            logging.error("--bucket is required when using --ingest")
            return

        try:
            from src.rag import ingest_document, compute_file_checksum as rag_checksum
        except ImportError:
            logging.error("RAG module not available. Install psycopg2: pip install psycopg2-binary")
            return

        try:
            # Use the cached transcription content
            file_checksum = rag_checksum(args.pdf_file) if os.path.exists(args.pdf_file) else None
            source_name = os.path.basename(args.pdf_file)

            result = ingest_document(
                text=transcription,
                bucket_id=args.bucket,
                source_file=source_name,
                file_checksum=file_checksum,
                metadata={"transcribed_at": cache_path},
                force=args.rag_force
            )

            if result["status"] == "success":
                print(f"✓ Ingested into RAG: {result['chunk_count']} chunks, ~{result['total_tokens']} tokens")
            elif result["status"] == "skipped":
                print(f"○ Already indexed (use --rag-force to re-ingest)")
            else:
                print(f"✗ RAG ingestion failed: {result.get('reason', 'unknown')}")

        except Exception as e:
            logging.error(f"RAG ingestion failed: {e}")


if __name__ == "__main__":
    main()
