#!/usr/bin/env python3
import sys
import os
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf, pages_per_chunk=50):
    if not os.path.exists(input_pdf):
        print(f"Error: File '{input_pdf}' not found.")
        sys.exit(1)

    # Open the PDF file
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    print(f"Total pages in {input_pdf}: {total_pages}")

    chunk_index = 1

    # Loop through the PDF pages in increments of pages_per_chunk
    for start in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        end = min(start + pages_per_chunk, total_pages)
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        
        # Create an output file name based on the input file name and chunk index
        base_name, ext = os.path.splitext(input_pdf)
        output_pdf = f"{base_name}_part{chunk_index}{ext}"
        
        # Write the chunk to a new PDF file
        with open(output_pdf, 'wb') as out_file:
            writer.write(out_file)
        
        print(f"Created: {output_pdf}")
        chunk_index += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 split_pdf.py input_pdf [pages_per_chunk]")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    
    # Optionally, you can specify a different chunk size
    pages_per_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    split_pdf(input_pdf, pages_per_chunk)

