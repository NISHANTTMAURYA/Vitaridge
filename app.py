from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import PyPDF2
import pytesseract
from PIL import Image
import io
from transformers import pipeline
from groq import Groq
import json

app = Flask(__name__, static_url_path='/static')
app.static_folder = 'static'

# Add this route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Define the upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize Groq client
client = Groq(
    api_key="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"
)

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

# Replace the existing summarizer pipeline with this function
def generate_summary(text):
    """Generate a medical report analysis using Groq LLM."""
    try:
        system_prompt = """You are an expert medical assistant specialized in analyzing medical reports. 
Your task is to:
1. Identify key medical findings
2. Extract vital measurements and lab results
3. Highlight any concerning values or abnormalities
4. Provide a clear, structured summary
5. Suggest potential follow-up actions if necessary

Format your response in the following structure:
{
    "key_findings": [list of main medical observations],
    "vital_signs": {relevant measurements},
    "lab_results": {significant values},
    "concerns": [list of abnormal or concerning values],
    "summary": "concise overview",
    "recommendations": [suggested follow-up actions]
}"""

        user_prompt = f"Please analyze this medical report and provide a structured analysis:\n\n{text}"

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=2048
        )

        # Extract and parse the response
        response = chat_completion.choices[0].message.content
        try:
            # Try to parse as JSON first
            structured_response = json.loads(response)
            return structured_response
        except json.JSONDecodeError:
            # If not valid JSON, return as is
            return {"summary": response}

    except Exception as e:
        print(f"[ERROR] Generating analysis: {str(e)}")
        return {"error": "Failed to analyze the medical report"}

# Modify the home route to serve index.html instead of mr.html
@app.route('/')
def home():
    return render_template('index.html')

# Add new routes for each service
@app.route('/mr')
def medical_report():
    return render_template('mr.html')

@app.route('/pg')
def prescription_generator():
    return render_template('pg.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/dc')
def doctor_consult():
    return render_template('dc.html')

@app.route('/dd')
def drug_data():
    return render_template('dd.html')

@app.route('/vr')
def vaccination_records():
    return render_template('vr.html')

@app.route('/upload', methods=['POST'])
def summarize():
    """Handle file upload and return analyzed medical report."""
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

        # Extract text based on file type
        if file_ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            text = extract_text_from_image(file_path)
        else:
            return jsonify({"error": "Unsupported file format. Upload PDF or image."}), 400

        if not text:
            return jsonify({"error": "No text found in the uploaded file."}), 400

        print(f"✅ Extracted text: {text[:100]}...")

        # Generate analysis using Groq
        analysis = generate_summary(text)
        print(f"✅ Generated Analysis: {json.dumps(analysis, indent=2)}")

        return jsonify(analysis)

    except Exception as e:
        print(f"🚨 Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add to app.py
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')

        system_prompt = """You are an expert medical AI assistant trained to conduct thorough medical conversations.

Your approach should be:

1. Initial Response:
   - Acknowledge the user's concern
   - Ask relevant follow-up questions to gather more context
   - Present questions one at a time to avoid overwhelming the user

2. Question Strategy:
   - Ask about symptoms duration and severity
   - Inquire about relevant medical history
   - Check for related symptoms
   - Ask about lifestyle factors that may be relevant
   - Verify if any treatments have been tried

3. Analysis and Response:
   - Synthesize the information provided
   - Explain medical terms in simple language
   - Highlight potential connections between symptoms
   - Suggest general wellness recommendations
   - Include relevant preventive measures

4. Safety Protocols:
   - Clearly state you are not replacing medical professionals
   - Advise seeking immediate medical attention for serious symptoms
   - Include appropriate medical disclaimers
   - Flag emergency situations immediately

5. Communication Style:
   - Maintain a professional yet empathetic tone
   - Use clear, simple language
   - Break down complex medical concepts
   - Be patient-centric and supportive
   - Show appropriate concern without causing alarm

Format your responses as:
1. Brief acknowledgment of the user's input
2. 1-2 relevant follow-up questions
3. Helpful information or guidance based on available context
4. Appropriate medical disclaimer when needed

Remember: Always maintain a conversational flow while gathering necessary medical information."""

        # Call Groq API with conversation context
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1024
        )

        response = chat_completion.choices[0].message.content
        return jsonify({"response": response})

    except Exception as e:
        print(f"[ERROR] Chat processing: {str(e)}")
        return jsonify({"error": "Failed to process your message"}), 500

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(debug=True)
