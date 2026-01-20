# plugins/pdfscribe_v2_plugin.py

from .plugin_interface import PluginInterface
from django.urls import path
from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import logging
import os
import io
import base64
import requests
from pdf2image import convert_from_bytes
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# API key loaded from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

class PDFScribeV2Plugin(PluginInterface):
    def initialize(self):
        print("PDFScribeV2Plugin initialized")

    def get_routes(self):
        return [
            path('pdfscribe_v2/', upload_form, name='pdfscribe_v2_upload'),
            path('pdfscribe_v2/upload', upload_file, name='pdfscribe_v2_process'),
            path('pdfscribe_v2/combine', combine_pages, name='pdfscribe_v2_combine'),
            path('pdfscribe_v2/admin', admin, name='pdfscribe_v2_admin'),
        ]

    def get_menu_items(self):
        return [{'name': 'PDFScribeV2', 'url': 'pdfscribe_v2_upload'}]

    parameters = {
        "prompt_text": """You are a skilled transcriber specializing in extracting text from poor-quality scanned document pages. Your task is to accurately transcribe all visible text and handle any handwritten annotations such as dates, checkmarks, or initials with special attention:
Annotations: Enclose handwritten annotations in curly brackets. For example, for handwritten initials, transcribe them as \{initials\}.
Uncertain Text: If you encounter any text that you are unsure about, highlight it in red to indicate uncertainty.
Formatting:
Tables: Use simple HTML tags to represent tables.
Bold Text: Use HTML tags to bold any text that appears bold in the original document.
Notes: Focus solely on transcription without adding any notes or commentary.""",
        "model": "gpt-4o",
        "max_tokens": 4000,
        "temperature": 0.7
    }

@login_required
def upload_form(request):
    return render(request, 'pdfscribe_v2/upload_v2.html')

@csrf_exempt
@login_required
def upload_file(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return HttpResponse("No file part", status=400)
        file = request.FILES['file']
        if file.name == '':
            return HttpResponse("No selected file", status=400)

        backstory = request.POST.get('backstory', '')

        if file:
            try:
                pdf_bytes = file.read()
                images = pdf_to_images(pdf_bytes)
                all_text = []
                page_count = 0

                def generate():
                    nonlocal all_text, page_count
                    yield "<html><body><a href='/'>Home</a><hr>"
                    for image in images:
                        page_count += 1
                        base64_image = encode_image(image)
                        text = image_to_text(base64_image, page_count, backstory)
                        all_text.append(text)
                        yield f"<div>{text}</div><hr>"
                    yield '''<div>
                                <button onclick="window.location.href='/'" class="btn btn-primary">Home</button>
                                <button onclick="combinePages(all_text)" class="btn btn-secondary">Combine Pages</button>
                             </div>
                             <script>
                                var all_text = ''' + str(all_text).replace("'", '"') + ''';
                                function combinePages(all_text) {
                                    fetch('/pdfscribe_v2/combine', {
                                        method: 'POST',
                                        headers: {
                                            'Content-Type': 'application/json',
                                        },
                                        body: JSON.stringify({pages: all_text}),
                                    })
                                    .then(response => response.text())
                                    .then(data => {
                                        document.body.innerHTML = data;
                                    })
                                    .catch(error => {
                                        console.error('Error:', error);
                                    });
                                }
                             </script>'''
                    yield "</body></html>"

                response = StreamingHttpResponse(generate(), content_type='text/html')
                response.headers["X-Accel-Buffering"] = "no"
                response.headers["Cache-Control"] = "no-cache"
                return response

            except Exception as e:
                logging.error(f"An error occurred: {e}")
                return HttpResponse(f"An error occurred: {e}", status=500)

    return HttpResponse("Invalid request method", status=405)

@csrf_exempt
@login_required
def combine_pages(request):
    if request.method == 'POST':
        data = request.json()
        pages = data.get('pages', [])

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        prompt_text = "Combine the following pages into one cohesive document:\n\n" + "\n\n".join(pages)

        payload = {
            "model": PDFScribeV2Plugin.parameters['model'],
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "max_tokens": PDFScribeV2Plugin.parameters['max_tokens'],
            "temperature": PDFScribeV2Plugin.parameters['temperature']
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            combined_text = response.json()['choices'][0]['message']['content']
            return HttpResponse(f"<html><body><div>{combined_text}</div><a href='/' class='btn btn-primary'>Home</a></body></html>")
        else:
            return HttpResponse("Failed to combine pages", status=500)

    return HttpResponse("Invalid request method", status=405)

@login_required
def admin(request):
    if request.method == 'POST':
        PDFScribeV2Plugin.parameters['prompt_text'] = request.POST['prompt_text']
        PDFScribeV2Plugin.parameters['model'] = request.POST['model']
        PDFScribeV2Plugin.parameters['max_tokens'] = int(request.POST['max_tokens'])
        PDFScribeV2Plugin.parameters['temperature'] = float(request.POST['temperature'])
        return redirect('pdfscribe_v2_admin')
    return render(request, 'pdfscribe_v2/admin.html', {'parameters': PDFScribeV2Plugin.parameters})

def pdf_to_images(pdf_bytes):
    logging.debug("Converting PDF to images")
    return convert_from_bytes(pdf_bytes)

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    logging.debug("Image encoded to base64")
    return encoded_image

def image_to_text(base64_image, page_count, backstory):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    prompt_text = (backstory + " " if backstory else "") + PDFScribeV2Plugin.parameters['prompt_text']

    payload = {
        "model": PDFScribeV2Plugin.parameters['model'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": PDFScribeV2Plugin.parameters['max_tokens'],
        "temperature": PDFScribeV2Plugin.parameters['temperature']
    }

    for attempt in range(5):
        logging.debug(f"Attempt {attempt + 1} to process image")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            logging.debug("Image processed successfully")
            return response.json()['choices'][0]['message']['content']
        else:
            logging.error(f"Attempt {attempt + 1} failed to process image: {response.json()}")
    return ""
