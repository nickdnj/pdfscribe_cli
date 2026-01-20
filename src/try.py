import re
import os

def clean_transcript(file_path, output_path):
    """Cleans the transcript file by removing artifacts and formatting it as a single HTML document."""
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return
    
    # Read the transcript file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove artifacts like ``` and --- Page Break ---
    content = re.sub(r'```+', '', content)  # Remove any triple backticks
    content = re.sub(r'---\s*Page Break\s*---', '', content)  # Remove page break markers
    
    # Remove excessive new lines
    content = re.sub(r'\n{2,}', '\n', content)  
    
    # Wrap the cleaned content in an HTML structure
    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>Unified Transcript</title>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    
    # Write the cleaned content to the output file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f"Cleaned transcript saved to {output_path}")

# Example usage
input_file = 'transcription.txt'  # Path to input file
output_file = 'cleaned_transcript.html'  # Output HTML file
clean_transcript(input_file, output_file)

