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
from typing import Dict, List, Union

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

def analyze_images(image_paths: List[str]) -> Dict[str, Union[Dict, str]]:
    """Generate detailed tags for the uploaded images using Gemini Vision"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    image_tags = {}
    
    for path in image_paths:
        try:
            with open(path, "rb") as img_file:
                image_data = img_file.read()
                
            prompt = [
                "Analyze this fabric image and provide detailed information about: "
                "material composition, color palette (including secondary colors), "
                "pattern characteristics, texture properties, and structural elements. "
                "Format as JSON with separate sections for each characteristic.",
                {"mime_type": "image/jpeg", "data": image_data}
            ]
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean the response text by removing extra formatting
            response_text = response_text.replace('```json\n', '').rstrip('```')
            
            # Validate response
            if not response_text:
                print(f"Warning: Empty response from Gemini for {path}")
                image_tags[os.path.basename(path)] = {"error": "Empty response from API"}
                continue
                
            try:
                # Parse the response text into a proper JSON structure
                parsed_response = json.loads(response_text)
                
                result = {
                    'color_palette': parsed_response.get('color_palette', {}).get('primary_color', 'unknown'),
                    'material_type': parsed_response.get('material_composition', {}).get('likely_material', 'unknown'),
                    'pattern': parsed_response.get('pattern_characteristics', {}).get('pattern_type', 'unknown'),
                    'texture': parsed_response.get('texture_properties', {}).get('visual_texture', 'unknown'),
                    'structural_elements': parsed_response.get('structural_elements', {}).get('details', [])
                }
                
                filename = os.path.basename(path)
                image_tags[filename] = result
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {path}: {str(e)}")
                print(f"Raw response: {response_text}")
                image_tags[os.path.basename(path)] = {"error": f"Invalid JSON response: {str(e)}"}
                
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                image_tags[os.path.basename(path)] = {"error": f"Processing error: {str(e)}"}
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            image_tags[os.path.basename(path)] = {"error": str(e)}
            
    return image_tags
    
def generate_initial_prompt(image_tags: Dict[str, Union[Dict, str]]) -> str:
    """Generate a creative prompt for fusing fabric elements into an upcycled design"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    fabric_descriptions = []
    for filename, tag_data in image_tags.items():
        try:
            if isinstance(tag_data, dict) and 'error' not in tag_data:
                # Use color_palette instead of colors
                colors = tag_data.get('color_palette', 'unknown')
                if isinstance(colors, list):
                    color_str = ', '.join(colors)
                else:
                    color_str = colors
                
                desc = f"{filename}: {color_str} {tag_data['material_type']} "
                desc += f"featuring {tag_data['pattern']} patterns and {tag_data['texture']} texture"
                fabric_descriptions.append(desc)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            fabric_descriptions.append(f"{filename}: (unreadable tags)")
    
        prompt = (
                "Design a practical, wearable garment that combines these materials in a simple, realstic and functional way. "
                "Focus on everyday wearability and comfort.DOES not need to be a artistic piece and DO NOT write a full exploration of themes or create an outfit"
                "Think about how these fabrics can work together in a straightforward, practical design please keep the concept realistic and not more than 300 words.\n\n"
                
                "Materials:\n" +
                "\n".join(fabric_descriptions) + "\n\n"
                
                "Design a single simple, practical garment NOT an outfit that combines these elements in a functional way"
                "MAX 150 TOKEN"
            )
    
    print("Getting creative concept from Gemini...")
    # Use lower temperature for more focused responses
    generation_config = {
        'temperature': 0.2,  # Lower temperature for more focused responses
        #  'max_tokens': 150  # need to understand this 
    }
    
    response = model.generate_content([prompt], generation_config=generation_config)
    creative_concept = response.text.strip()
    return creative_concept

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
        
        # Store results in session
        session['result_image_url'] = result['image_url']
        session['generation_text'] = result.get('generation_text', '')
        
        # Redirect to result page
        return redirect(url_for('result'))
    
    return render_template(
        'refine.html',
        image_paths=[f"/static/uploads/{session['user_id']}/{os.path.basename(path)}" 
                    for path in session.get('image_paths', [])],
        image_tags=session.get('image_tags', {}),
        base_prompt=session.get('base_prompt', '')
    )

@app.route('/result', methods=['GET'])
def result():
    """Result page with generated design"""
    if 'refined_prompt' not in session:
        return redirect(url_for('index'))
    
    return render_template(
        'result.html',
        prompt=session['refined_prompt'],
        image_url=session.get('result_image_url', ''),
        generation_text=session.get('generation_text', ''),
        image_paths=[f"/static/uploads/{session['user_id']}/{os.path.basename(path)}" 
                    for path in session.get('image_paths', [])]
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

    