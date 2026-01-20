#!/usr/bin/env python3
import argparse
import os
import io
import re
import base64
import logging
import requests
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env (if present)
load_dotenv()

# Retrieve your API key from the environment
API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
if API_KEY in ("INSERT API KEY", "your_api_key_here"):
    logging.warning("API_KEY is not set. Please set it in your environment variable OPENAI_API_KEY.")

# Global parameters for the prompt and API call
PARAMETERS = {
    "prompt_text": (
        "You are a skilled transcriber specializing in extracting text from poor-quality scanned document pages. "
        "Your task is to accurately transcribe all visible text and handle any handwritten annotations such as dates, "
        "checkmarks, or initials with special attention:\n"
        "Annotations: Enclose handwritten annotations in curly brackets. For example, for handwritten initials, transcribe them as {initials}.\n"
        "Uncertain Text: If you encounter any text that you are unsure about, highlight it in red to indicate uncertainty.\n"
        "Formatting:\n"
        "Tables: Use simple HTML tags to represent tables.\n"
        "Bold Text: Use HTML tags to bold any text that appears bold in the original document.\n"
        "Notes: Focus solely on transcription without adding any notes or commentary."
    ),
    "model": "gpt-4o",
    "max_tokens": 4000,
    "temperature": 0.7
}

def clean_filename(filename):
    """
    Remove spaces and special characters from a filename (without extension).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r'[^A-Za-z0-9]+', '', base)
    return cleaned

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

def save_image(image, base_name, page_number, images_dir):
    """Save the PIL image to disk inside images_dir using the base name and page number."""
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_filename = f"{base_name}_page_{page_number:03d}.png"
    image_path = os.path.join(images_dir, image_filename)
    image.save(image_path, format="PNG")
    logging.info(f"Saved image for page {page_number} at {image_path}")
    return image_path

def image_to_text(base64_image, page_number, backstory):
    """
    Send the encoded image along with a prompt (and optional backstory)
    to GPT-4o via the API and return the transcribed text.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    prompt_text = (backstory + " " if backstory else "") + PARAMETERS['prompt_text']

    payload = {
        "model": PARAMETERS['model'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": PARAMETERS['max_tokens'],
        "temperature": PARAMETERS['temperature']
    }

    for attempt in range(5):
        logging.info(f"Processing page {page_number}: Attempt {attempt + 1}")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            logging.info(f"Page {page_number} processed successfully.")
            return response.json()['choices'][0]['message']['content']
        else:
            logging.error(f"Attempt {attempt + 1} failed for page {page_number}: {response.json()}")
    logging.error(f"Page {page_number} failed after 5 attempts.")
    return f"[Error processing page {page_number}]"

def clean_transcription(text):
    """
    Remove unwanted artifacts (e.g. code block markers like ```html or ```) from the transcribed text.
    """
    cleaned = re.sub(r"```[^\n]*\n?", "", text)
    return cleaned.strip()

def generate_html_for_pdf(transcriptions, image_paths, human_readable, output_html):
    """
    Generate an HTML page for a single PDF document.
    This version does not include any per-document navigation.
    """
    pages_html = ""
    for i, (trans_text, img_path) in enumerate(zip(transcriptions, image_paths), start=1):
        cleaned_text = clean_transcription(trans_text)
        pages_html += f"""
<div class="page" id="page-{i}">
  <h2>Page {i}</h2>
  <div class="transcription">
    {cleaned_text}
  </div>
  <div class="image-link">
    <a href="{img_path}" target="_blank">View Original Image</a>
  </div>
</div>
"""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{human_readable} - Transcribed Document</title>
  <link rel="icon" type="image/png" href="favicon.png">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
    }}
    .page {{
      margin-bottom: 50px;
      padding-bottom: 20px;
      border-bottom: 1px solid #ccc;
    }}
    .image-link a {{
      text-decoration: none;
      color: #0066cc;
    }}
    .image-link a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <h1>{human_readable} - Transcribed Document</h1>
  {pages_html}
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"Document page generated: {output_html}")

def generate_home_page(global_outdir, site_title, output_file="home.html"):
    """
    Generate a home page that describes the website.
    Explains that the transcriptions were produced using GPT-4o, and:
      - Red text indicates uncertain transcription.
      - Handwritten annotations are enclosed in curly brackets.
    The custom site title (provided via -t/--title) is shown above the welcome message.
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Home - Transcribed {site_title} Website</title>
  <link rel="icon" type="image/png" href="favicon.png">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      line-height: 1.6;
    }}
    h2 {{
      color: #333;
      margin-bottom: 10px;
    }}
    h1 {{
      color: #0066cc;
    }}
    .explanation {{
      margin-top: 20px;
    }}
    .explanation p {{
      margin-bottom: 15px;
    }}
    .note {{
      color: red;
      font-weight: bold;
    }}
    .annotation {{
      font-style: italic;
    }}
  </style>
</head>
<body>
  <h2>{site_title}</h2>
  <h1>Welcome to the Transcribed Documents Website</h1>
  <div class="explanation">
    <p>This website displays transcriptions of scanned documents that have been processed using an advanced transcription model (GPT-4o).</p>
    <p><strong>How It Works:</strong> The model transcribes all visible printed text from each scanned page. Handwritten annotations—such as dates, checkmarks, or initials—are enclosed in curly brackets. For example, a handwritten "J.D." appears as <span class="annotation">{{J.D.}}</span>.</p>
    <p>If the model is uncertain about any part of the text, that portion is highlighted in <span class="note">red</span>. This red highlight serves as a cue for you to refer to the original scanned image for verification.</p>
    <p>Additionally, tables and bold text are formatted using simple HTML tags to retain their structure and emphasis.</p>
    <p>Use the sidebar on the left to navigate among the documents. Click "Home" at any time to view this information again.</p>
  </div>
</body>
</html>
"""
    home_path = os.path.join(global_outdir, output_file)
    with open(home_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"Home page generated: {home_path}")
    return output_file

def generate_global_index(documents, global_outdir, output_file="index.html"):
    """
    Generate a global index HTML page with a fixed left-hand sidebar.
    The sidebar includes a "Home" link plus a list of document titles.
    When a title is clicked, its document is loaded into an iframe.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Transcribed Documents Index</title>
  <link rel="icon" type="image/png" href="favicon.png">
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
    }
    .sidebar {
      background: #f0f0f0;
      width: 250px;
      height: 100vh;
      position: fixed;
      overflow-y: auto;
      padding: 20px;
      box-sizing: border-box;
      border-right: 1px solid #ccc;
    }
    .sidebar h2 {
      margin-top: 0;
    }
    .sidebar ul {
      list-style-type: none;
      padding: 0;
    }
    .sidebar li {
      margin: 10px 0;
    }
    .sidebar a {
      text-decoration: none;
      color: #0066cc;
      display: block;
    }
    .sidebar a:hover {
      text-decoration: underline;
    }
    .content {
      margin-left: 270px;
      padding: 20px;
    }
    iframe {
      width: 100%;
      height: 90vh;
      border: none;
    }
  </style>
</head>
<body>
  <div class="sidebar">
    <h2>Documents</h2>
    <ul>
      <li><a href="home.html" target="content_frame">Home</a></li>
"""
    for human_readable, html_filename in documents:
        html_content += f'      <li><a href="{html_filename}" target="content_frame">{human_readable}</a></li>\n'
    html_content += """    </ul>
  </div>
  <div class="content">
    <iframe name="content_frame" src="home.html"></iframe>
  </div>
</body>
</html>
"""
    index_path = os.path.join(global_outdir, output_file)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"Global index generated: {index_path}")

def process_pdf(pdf_path, backstory, global_outdir, folder_prefix=None):
    """
    Process a single PDF file and generate its HTML page.
    Returns a tuple: (human_readable, html_filename) for inclusion in the global index.
    """
    human_readable = os.path.splitext(os.path.basename(pdf_path))[0]
    base_clean = clean_filename(pdf_path)
    if folder_prefix:
        html_filename = f"{folder_prefix}_{base_clean}.html"
    else:
        html_filename = f"{base_clean}.html"
    output_html = os.path.join(global_outdir, html_filename)
    logging.info(f"Processing PDF: {pdf_path} -> {output_html}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    try:
        images = pdf_to_images(pdf_bytes)
    except Exception as e:
        logging.error(f"Failed to convert PDF to images for {pdf_path}: {e}")
        return None

    transcriptions = []
    image_paths = []
    images_dir = os.path.join(global_outdir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for i, image in enumerate(tqdm(images, desc=f"Processing {human_readable}", unit="page")):
        page_number = i + 1
        img_full_path = save_image(image, clean_filename(pdf_path), page_number, images_dir)
        img_rel_path = os.path.relpath(img_full_path, global_outdir)
        image_paths.append(img_rel_path)
        base64_img = encode_image(image)
        text = image_to_text(base64_img, page_number, backstory)
        transcriptions.append(text)

    generate_html_for_pdf(transcriptions, image_paths, human_readable, output_html)
    return (human_readable, html_filename)

def main():
    parser = argparse.ArgumentParser(
        description="Process PDF(s) with GPT-4o to generate transcribed websites. Input can be a single PDF file or a folder containing PDFs."
    )
    parser.add_argument("input_path", help="Path to a PDF file or a folder of PDFs")
    parser.add_argument("-b", "--backstory", help="Optional backstory text to prepend to the prompt", default="")
    parser.add_argument("-o", "--outdir", help="Output directory for generated files", default="output")
    parser.add_argument("-t", "--title", help="Custom title for the website", default="Documents")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    global_outdir = args.outdir
    if not os.path.exists(global_outdir):
        os.makedirs(global_outdir)

    pdf_files = []
    folder_prefix = None

    if os.path.isdir(args.input_path):
        pdf_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                     if f.lower().endswith(".pdf")]
        if not pdf_files:
            logging.error(f"No PDF files found in folder '{args.input_path}'.")
            return
        folder_prefix = clean_filename(os.path.basename(os.path.normpath(args.input_path)))
    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith(".pdf"):
        pdf_files = [args.input_path]
    else:
        logging.error("Input must be a PDF file or a folder containing PDF files.")
        return

    processed_docs = []
    for pdf in pdf_files:
        result = process_pdf(pdf, args.backstory, global_outdir, folder_prefix)
        if result:
            processed_docs.append(result)

    # Generate the home page using the custom title provided via -t/--title.
    generate_home_page(global_outdir, site_title=args.title, output_file="home.html")
    
    # Generate the global index.
    index_path = os.path.join(global_outdir, "index.html")
    generate_global_index(processed_docs, global_outdir, output_file="index.html")
    
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
