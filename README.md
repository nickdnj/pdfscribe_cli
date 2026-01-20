# PDFScribe CLI

**Transform scanned PDFs into searchable, structured Markdown using Claude's vision capabilities.**

PDFScribe CLI is a command-line tool that converts image-based PDFs (scanned documents, inspection reports, legacy files) into detailed Markdown transcriptions. It uses Anthropic's Claude Sonnet 4 vision model to accurately transcribe text, describe images, and preserve document structure.

## Features

- **Vision-based transcription** - Uses Claude Sonnet 4 to read scanned/image PDFs
- **Intelligent caching** - Stores transcriptions next to source files, validates by checksum
- **Google Drive integration** - Download PDFs and upload transcriptions automatically
- **Handwriting support** - Captures handwritten annotations in `{curly brackets}`
- **Table extraction** - Converts tables to HTML format
- **Multi-page processing** - Handles documents of any length with progress tracking
- **Retry logic** - Automatic retry with exponential backoff for API errors

## Installation

### Prerequisites

- Python 3.8+
- [Poppler](https://poppler.freedesktop.org/) (for PDF to image conversion)
- Anthropic API key

### macOS

```bash
# Install poppler
brew install poppler

# Clone the repository
git clone https://github.com/yourusername/pdfscribe_cli.git
cd pdfscribe_cli

# Install Python dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Dependencies

```
pdf2image
python-dotenv
requests
tqdm
google-api-python-client  # For Google Drive integration
google-auth-oauthlib      # For Google Drive authentication
```

## Quick Start

### Basic Usage

```bash
# Transcribe a local PDF (output saved next to source)
python pdfscribe_cli.py document.pdf

# Output: document-transcribed.md
```

### Google Drive Workflow

```bash
# Download from Drive, transcribe, upload MD back to Drive
python pdfscribe_cli.py --gdrive 1xtoBO7vjnOfNoXmdDn7w3sg-Ds1zHmdt

# The transcription is automatically uploaded next to the source PDF
```

### Command Line Options

```
usage: pdfscribe_cli.py [-h] [-o OUTPUT] [-b BACKSTORY] [-v] [-m MODEL]
                        [--no-cache] [--force] [--gdrive FILE_ID]
                        [--gdrive-upload] [--gdrive-folder FOLDER_ID]
                        [--work-dir WORK_DIR]
                        [pdf_file]

Arguments:
  pdf_file              Path to the PDF file (or use --gdrive)

Options:
  -o, --output          Custom output file path
  -b, --backstory       Context to help with transcription
  -v, --verbose         Enable verbose logging
  -m, --model           Claude model (default: claude-sonnet-4-20250514)
  --no-cache            Disable caching
  --force               Force re-transcription (ignore cache)
  --gdrive FILE_ID      Google Drive file ID to process
  --gdrive-upload       Upload transcription to Google Drive
  --gdrive-folder ID    Target folder for upload (auto-detected if using --gdrive)
  --work-dir            Working directory for downloads (default: /tmp/pdfscribe)
```

## Caching

PDFScribe uses intelligent caching to avoid redundant API calls:

- **Cache location**: `{filename}-transcribed.md` next to source PDF
- **Validation**: SHA256 checksum of source PDF
- **Auto-invalidation**: Re-transcribes when source changes
- **Performance**: ~350x faster on cache hit

### Cache Metadata

Each transcription includes a metadata header:

```markdown
<!--
PDFScribe Cache
Source: /path/to/document.pdf
Original: document.pdf
Transcribed: 2026-01-20T17:15:18Z
Model: claude-sonnet-4-20250514
Pages: 12
Checksum: 8bd4cce9e0853eadedf8205f40bff4f1f077db7cdaedaa352d79af73401f0cbc
-->

[transcription content...]
```

## Google Drive Setup

To enable Google Drive integration:

1. Create OAuth credentials in Google Cloud Console
2. Save credentials to `~/.config/mcp-gdrive/gcp-oauth.keys.json`
3. Run the authentication setup:

```bash
python setup_gdrive_auth.py
```

This grants PDFScribe permission to:
- Read files from Google Drive
- Create/update transcription files

## Output Format

Transcriptions are structured Markdown with:

- Page breaks marked with `--- Page Break ---`
- Handwritten annotations in `{curly brackets}`
- Tables converted to HTML `<table>` format
- Bold text preserved with HTML `<b>` tags
- Uncertain text flagged appropriately

### Example Output

```markdown
**Building 7 Crawl Space Inspection**

Inspection Date: 08/21/2025

**Property:** Wharfside Dr, Monmouth Beach, NJ 07750

**Interior**

Crawlspace condition: {good} ✓

| Area | Condition | Notes |
|------|-----------|-------|
| Foundation | Good | No cracks observed |
| Insulation | Fair | Some areas need attention |

--- Page Break ---

**Photos**

[Image 1: Shows crawlspace entrance with wooden access door...]
```

## Performance

| Document | Pages | First Run | Cache Hit |
|----------|-------|-----------|-----------|
| 1-page PDF | 1 | ~6s | 0.2s |
| 12-page PDF | 12 | ~70s | 0.2s |
| 17-page PDF | 17 | ~90s | 0.2s |

Average: **~5-6 seconds per page** for transcription.

## MCP Server Integration

PDFScribe can be used as an MCP (Model Context Protocol) server for Claude Code and other AI assistants. See the `mcp-servers/images/pdfscribe/` directory for:

- Docker configuration
- MCP server wrapper
- Claude Code integration

## Use Cases

- **Property inspections** - Transcribe scanned inspection reports
- **Legal documents** - Make legacy contracts searchable
- **Medical records** - Extract text from scanned forms
- **Historical archives** - Digitize old documents
- **Research papers** - Process scanned academic papers

## Troubleshooting

### "ANTHROPIC_API_KEY environment variable is required"

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### "Failed to convert PDF to images"

Install poppler:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### Google Drive permission errors

Re-run authentication to get write access:
```bash
python setup_gdrive_auth.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude's vision capabilities
- [pdf2image](https://github.com/Belval/pdf2image) for PDF conversion
- [Poppler](https://poppler.freedesktop.org/) for PDF rendering

---

**Built with Claude** - This tool was developed in collaboration with Claude Code.
