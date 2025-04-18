#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install required packages
get_ipython().system('pip install diffusers transformers xformers git+https://github.com/huggingface/accelerate.git')
get_ipython().system('pip install opencv-contrib-python controlnet_aux google-generativeai gradio')


# In[ ]:


# Import required libraries
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import numpy as np
import cv2
from google.colab import files
from io import BytesIO
import google.generativeai as genai
import gradio as gr


# In[ ]:


from google.colab import userdata
userdata.get('HF_TOKEN')


# In[ ]:


key = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=key)


# In[ ]:


test_prompt = "best tupac one liner."
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content([test_prompt])
print("Gemini response:", response.text.strip())


# In[ ]:


# Enable GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize ControlNet model with optimizations
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_softedge",
    torch_dtype=torch.float16
)

# Initialize pipeline with optimizations
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Configure pipeline for optimal performance
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


# In[ ]:


import tempfile
import shutil

from io import BytesIO
from google.generativeai.types import content_types
from PIL import ImageFilter

def save_uploaded_files(file_objs):
    image_paths = []

    try:
        for file_obj in file_objs:
            temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
            with open(file_obj.name, "rb") as src:
                with open(temp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            image_paths.append(temp_path)

        print("Saved all files:", image_paths)
        return image_paths

    except Exception as e:
        print("Error saving files:", str(e))
        return []

def analyze_from_uploaded_files(file_objs):
    try:
        image_paths = save_uploaded_files(file_objs)


        # Analyze images via Gemini
        analysis = analyze_images(image_paths)
        print("Got analysis:", analysis)
        control_image = process_and_blend_images(image_paths)

              # Ensure control_image is a PIL.Image (if it's not already)
        if not isinstance(control_image, Image.Image):
            print("converting to Pill.IMage")
            control_image = Image.fromarray(control_image)  # Converts numpy array to PIL.Image

        return json.dumps(analysis, indent=2), analysis, control_image

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", {}

    finally:
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)



# 🧠 Analyze images using Gemini Vision
import os
import uuid
from typing import Dict, List, Union
import json

def analyze_images(image_paths: List[str]) -> Dict[str, Union[Dict, str]]:
    """Generate detailed tags for the uploaded images using Gemini Vision"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    image_tags = {}
    print('hey there from analyze_images')
    for path in image_paths:
        print('path' + path)
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

# 🧵 Generate design prompt from fabric analysis
def create_design_prompt(image_tags, control_image):
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


        # 👉 Encode the provided control image (already a PIL.Image)

        print("setting up control_image_data")
        if not isinstance(control_image, Image.Image):
            print("Expected PIL.Image, got", type(control_image))
            return "Error: Control image must be a PIL image"
        prompt = "\n".join([
            "You are a designer. The following image is a guide only for general structure, not literal content.",
            "The goal is to design an original, cohesive garment combining the materials below.",
            "Think about how these fabrics can work together in a straightforward, practical design please keep the concept realistic and not more than 300 words.\n\n",
            "Materials:\n" + "\n".join(fabric_descriptions),
            "Design a single simple, practical garment NOT an outfit that combines these elements in a functional way\nMAX 100 TOKEN"
        ])

    print("Getting creative concept from Gemini...")
        # Use lower temperature for more focused responses
    generation_config = {
        'temperature': 0.2,  # Lower temperature for more focused responses
        #  'max_tokens': 150  # need to understand this
    }

    response = model.generate_content([prompt, control_image], generation_config=generation_config)
    creative_concept = response.text.strip()
    return creative_concept

# 🎨 Generate final image
def final_design_wrapper(prompt, files):
    """
    Wraps your generate_final_design to handle Gradio File objects.
    """
    image_paths = save_uploaded_files(files)  # same method as before

      # 2. Blend them into a control image
    control_image = process_and_blend_images(image_paths)


    return generate_design(prompt, [control_image])[0]


  # Helper function for image processing
def process_and_blend_images(image_paths, size=(512, 512)):
    """
    Process multiple images and blend their Canny edge maps into one control image.
    """
    edge_accumulator = None
    for path in image_paths:
        image = Image.open(path).resize(size).convert("RGB")
        np_image = np.array(image)
        edges = cv2.Canny(np_image, 100, 200)
        edges_rgb = np.stack([edges]*3, axis=-1)  # Convert to RGB-like array

        if edge_accumulator is None:
            edge_accumulator = edges_rgb.astype(np.float32)
        else:
            edge_accumulator += edges_rgb.astype(np.float32)

    # Average edges
    averaged_edges = (edge_accumulator / len(image_paths)).astype(np.uint8)
    control_image = Image.fromarray(averaged_edges).convert("RGB")
    return control_image

def generate_design(prompt, control_images, num_steps=20, guidance_scale=5):
    """
    Generate designs using the ControlNet pipeline for multiple control images.

    Args:
        prompt (str): The text prompt to guide generation.
        control_images (List[PIL.Image]): List of processed control images.
        num_steps (int): Number of inference steps.
        guidance_scale (float): Classifier-free guidance scale.

    Returns:
        List[PIL.Image]: Generated images for each control image.
    """
    results = []
    for i, control_image in enumerate(control_images):
        print(f"Generating image {i+1}/{len(control_images)}...")
        soft_control = control_image.filter(ImageFilter.GaussianBlur(radius=3.0))
        result = pipe(
            prompt=prompt,
            controlnet_conditioning_scale=0.4,
            image=soft_control,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        results.append(result)
    return results


# In[ ]:


import gradio as gr
from PIL import Image
# Gradio interface

def load_images_from_files(files):
    return [Image.open(file) for file in files]


with gr.Blocks() as demo:
    gr.Markdown("## 🧥 Fabric Fusion Designer")

    with gr.Row():
        image_input = gr.File(
            file_types=["image"],
            label="Upload Fabric Images",
            file_count="multiple"
        )
        analyze_button = gr.Button("Analyze Fabrics 🧠")

          # Configure the gallery to display smaller images
    uploaded_images_display = gr.Gallery(
        label="📸 Uploaded Fabric Images",
        columns=3,               # Display 3 images per row
        height="300px",          # Set the height of the gallery
        object_fit="contain"     # Ensure images are fully visible without cropping
    )

        # Update gallery when new files are uploaded
    image_input.change(
        fn=load_images_from_files,
        inputs=image_input,
        outputs=uploaded_images_display
    )

    analysis_output = gr.Textbox(label="📝 Fabric Analysis JSON", lines=15)
    control_image_display = gr.Image(label="🧭 Control Image (softedge map)", type="pil")

    prompt_state = gr.State()
    analysis_state = gr.State()
    image_path_state = gr.State()  # <- New!
    control_image_state = gr.State()  # <- New!

    def load_images_from_files(files):
        return [Image.open(file) for file in files]

        # Step 1: Analyze fabrics and generate control image
    analyze_button.click(
        fn=analyze_from_uploaded_files,
        inputs=image_input,
        outputs=[analysis_output, analysis_state, control_image_display]
    ).then(
        lambda x: x,  # Pass control image to state
        inputs=control_image_display,
        outputs=control_image_state
    )

    generate_prompt_button = gr.Button("Generate Prompt from Analysis 🧶")
    prompt_output = gr.Textbox(label="🧵 Generated Design Prompt", lines=4)

    generate_prompt_button.click(
        fn=create_design_prompt,
        inputs=[analysis_state, control_image_state],
        outputs=[prompt_output]
    ).then(
        fn=lambda x: x,
        inputs=prompt_output,
        outputs=prompt_state
    )

    design_button = gr.Button("Generate Final Garment Design 🎨")
    design_output = gr.Image(label="Final Design", type="pil")
    design_button.click(
        fn=final_design_wrapper,
        inputs=[prompt_state, image_input],
        outputs=design_output
    )

demo.launch(debug=True)

