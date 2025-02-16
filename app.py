from flask import Flask, request, jsonify
import os
import PyPDF2
import pytesseract
from PIL import Image
import io
from transformers import pipeline

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the text summarization model
summarizer = pipeline("summarization")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print("[ERROR] Extracting text from PDF:", str(e))
        raise e
    return text

def extract_text_from_image(image_path):
    """Extract text from an image file using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print("[ERROR] Extracting text from image:", str(e))
        raise e

def generate_summary(text):
    """Generate a summary using a transformer-based model."""
    try:
        if len(text) < 50:
            return "The extracted text is too short for summarization."
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print("[ERROR] Generating summary:", str(e))
        return "Failed to generate summary."

@app.route('/')
def home():
    return "Welcome to the Medical Report Summarization API! Use /upload to upload your file."

@app.route('/upload', methods=['POST'])
def summarize():
    """Handle file upload and return a summarized text."""
    if "reportFile" not in request.files:
        print("🚨 No file part in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["reportFile"]

    if file.filename == "":
        print("🚨 No selected file")
        return jsonify({"error": "No selected file"}), 400

    print(f"✅ Received file: {file.filename}")  # Debugging statement

    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        file.save(file_path)
        print(f"✅ File saved to: {file_path}")

        if file_ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            text = extract_text_from_image(file_path)
        else:
            print("🚨 Unsupported file format")
            return jsonify({"error": "Unsupported file format. Upload PDF or image."}), 400

        if not text:
            print("🚨 No text found in file")
            return jsonify({"error": "No text found in the uploaded file."}), 400

        print(f"✅ Extracted text: {text[:100]}...")  # Print first 100 characters

        summary = generate_summary(text)
        print(f"✅ Generated Summary: {summary}")

        return jsonify({"summary": summary})

    except Exception as e:
        print(f"🚨 Error processing file: {str(e)}")  # Debugging statement
        return jsonify({"error": "Failed to process the file. Please try again."}), 500

        # Generate summary
        summary = generate_summary(text)
        print("[INFO] Summary:", summary)

        return jsonify({"summary": summary})

    except Exception as e:
        print("[ERROR] Processing file:", str(e))  # Debugging
        return jsonify({"error": "Failed to process the file. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
