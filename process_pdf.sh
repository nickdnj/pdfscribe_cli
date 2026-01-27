#!/bin/bash -x

# Input and output directories
INPUT_DIR="ExecMinutes"
OUTPUT_DIR="ExecMinute_oututs"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Find and process PDF files
find "$INPUT_DIR" -type f -name "*.pdf" | while IFS= read -r pdf_file; do
    # Extract the filename without the extension
    filename=$(basename -- "$pdf_file")
    filename_no_ext="${filename%.*}"

    # Define the output file name
    output_file="$OUTPUT_DIR/$filename_no_ext.html"

    # Display the command before executing
    echo "Running: python3 pdfscribe_cli.py \"$pdf_file\" -o \"$output_file\""

    # Run the actual command
    python3 pdfscribe_cli.py "$pdf_file" -o "$output_file"

    echo "Processed: $pdf_file → Saved as $output_file"
done

