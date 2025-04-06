# app.py
import os
import uuid
import base64
import mimetypes
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from google.cloud import storage
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from helpers.dalle import generate_dalle_image
import json

# Load environment variables
load_dotenv()

# Configure Google Cloud and Gemini API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/iparvez/.config/gcloud/application_default_credentials.json"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload and generated directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket_name = os.getenv("GCS_BUCKET_NAME")
bucket = storage_client.bucket(bucket_name)

# Helper functions
def generate_user_id():
    """Generate a unique user ID for the session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def upload_to_gcs(file_path, user_id, filename):
    """Upload file to Google Cloud Storage"""
    destination_blob_name = f"uploads/{user_id}/{filename}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    return f"gs://{bucket_name}/{destination_blob_name}"

def analyze_images(image_paths):
    """Generate tags for the uploaded images using Gemini Vision"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    image_tags = {}
    for path in image_paths:
        image_data = open(path, "rb").read()
        response = model.generate_content([
            "Analyze this fabric image and provide tags for: material type, color, pattern, and texture. Format as JSON.",
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        # For simplicity we're assuming the response is well-formed
        # In production, you would want more robust parsing
        try:
            tags = response.text
            filename = os.path.basename(path)
            image_tags[filename] = tags
        except Exception as e:
            image_tags[os.path.basename(path)] = str(e)
    
    return image_tags

def generate_initial_prompt(image_tags):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Convert stringified JSON values into real dicts and format nicely
    fabric_descriptions = []
    for filename, tag_str in image_tags.items():
        try:
            print(f"Processing file: {filename}")  # Print filename for debugging
            print(f"Tag string: {tag_str}")  # Print the raw tag string to see what it looks like
            tags = json.loads(tag_str)  # Attempt to parse the JSON
            desc = f"{filename}: {tags['color']} {tags['material_type']} with a {tags['pattern']} pattern and {tags['texture']} texture"
            fabric_descriptions.append(desc)
        except Exception as e:
            print(f"Error processing {filename}: {e}")  # Print the error message
            fabric_descriptions.append(f"{filename}: (unreadable tags)")

    formatted_tags = "\n".join(fabric_descriptions)
        
    input_text = (
            "You're a fashion design assistant. "
            "Based on the following fabric descriptions, generate a short, creative design idea (2-4 sentences). "
            "Do NOT write a full design brief, assignment, or requirements list. Just a quick idea to inspire the user.\n\n"
            f"Fabric descriptions: {formatted_tags}"
        )
    print(input_text)
    response = model.generate_content(input_text)
    return response.text

def generate_refined_image(prompt):
    """Generate a design image based on the refined prompt using Gemini 2.0 Flash"""
    # Generate unique filename for this generation
    file_id = str(uuid.uuid4())
    output_dir = app.config['GENERATED_FOLDER']
    

    try:
        image_url, error = generate_dalle_image(prompt, output_dir)

        return {
            "prompt": prompt,
            "image_url": image_url,
            "generation_text": "",  # DALL·E doesn't return text
            "error": error
        }

    except Exception as e:
        app.logger.error(f"Image generation error: {str(e)}")
        return {
            "prompt": prompt,
            "image_url": "/static/placeholder-image.jpg",
            "error": str(e)
        }

# Routes
@app.route('/')
def index():
    """Landing page"""
    generate_user_id()  # Ensure user has an ID
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle fabric image uploads"""

    if request.form.get('new_session') == 'true':
        session.clear()
    
    user_id = generate_user_id()
    
    # Create user directory if it doesn't exist
    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    uploaded_files = request.files.getlist('fabric_images')
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    local_paths = []
    gcs_paths = []
    
    for file in uploaded_files:
        if file:
            filename = secure_filename(file.filename)
            local_path = os.path.join(user_dir, filename)
            file.save(local_path)
            local_paths.append(local_path)
            
            # Upload to Google Cloud Storage
            gcs_path = upload_to_gcs(local_path, user_id, filename)
            gcs_paths.append(gcs_path)
    
    # Analyze images and store results in session
    image_tags = analyze_images(local_paths)
    session['image_tags'] = image_tags
    session['image_paths'] = local_paths
    session['gcs_paths'] = gcs_paths
    
    # Generate initial prompt
    initial_prompt = generate_initial_prompt(image_tags)
    session['base_prompt'] = initial_prompt
    
    return redirect(url_for('refine_prompt'))

@app.route('/refine', methods=['GET', 'POST'])
def refine_prompt():
    """Prompt refinement page"""
    if 'image_tags' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        user_input = request.form.get('refinement', '')
        base_prompt = session.get('base_prompt', '')
        
        # Refine the prompt
        refined_prompt = f"{base_prompt} Style notes: {user_input}"
        session['refined_prompt'] = refined_prompt
        
        # Generate design based on refined prompt
        result = generate_refined_image(refined_prompt)
        
        return render_template(
            'results.html',
            prompt=refined_prompt,
            image_url=result['image_url'],
            generation_text=result.get('generation_text', ''),
            image_paths=[f"/static/uploads/{session['user_id']}/{os.path.basename(path)}" for path in session.get('image_paths', [])]
        )
    
    return render_template(
        'refine.html',
        image_paths=[f"/static/uploads/{session['user_id']}/{os.path.basename(path)}" for path in session.get('image_paths', [])],
        image_tags=session.get('image_tags', {}),
        base_prompt=session.get('base_prompt', '')
    )

@app.route('/api/regenerate', methods=['POST'])
def regenerate_design():
    """API endpoint to regenerate design with modified prompt"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    result = generate_refined_image(prompt)
    
    # Return all the information from the generation
    return jsonify({
        "prompt": result.get("prompt"),
        "image_url": result.get("image_url"),
        "generation_text": result.get("generation_text", ""),
        "error": result.get("error", None)
    })

if __name__ == '__main__':
    app.run(debug=True)

    