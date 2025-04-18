# app.py

import os
import json
import uuid
import shutil
import tempfile
from io import BytesIO
from typing import Dict, List, Union

import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
import gradio as gr

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import google.generativeai as genai

# Load secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_softedge",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# ========== UTILITIES ==========

def save_uploaded_files(file_objs):
    image_paths = []
    for file_obj in file_objs:
        temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        with open(file_obj.name, "rb") as src, open(temp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        image_paths.append(temp_path)
    return image_paths

def process_and_blend_images(image_paths, size=(512, 512)):
    edge_accumulator = None
    for path in image_paths:
        image = Image.open(path).resize(size).convert("RGB")
        np_image = np.array(image)
        edges = cv2.Canny(np_image, 100, 200)
        edges_rgb = np.stack([edges]*3, axis=-1)
        if edge_accumulator is None:
            edge_accumulator = edges_rgb.astype(np.float32)
        else:
            edge_accumulator += edges_rgb.astype(np.float32)
    averaged_edges = (edge_accumulator / len(image_paths)).astype(np.uint8)
    return Image.fromarray(averaged_edges).convert("RGB")

def analyze_images(image_paths: List[str]) -> Dict[str, Union[Dict, str]]:
    model = genai.GenerativeModel('gemini-2.0-flash')
    image_tags = {}
    for path in image_paths:
        try:
            with open(path, "rb") as img_file:
                image_data = img_file.read()

            prompt = [
                "Analyze this fabric image and provide detailed information about: "
                "material composition, color palette, pattern characteristics, texture properties, and structural elements.",
                {"mime_type": "image/jpeg", "data": image_data}
            ]
            response = model.generate_content(prompt)
            response_text = response.text.strip().replace("```json\n", "").rstrip("```")
            parsed_response = json.loads(response_text)

            result = {
                'color_palette': parsed_response.get('color_palette', {}).get('primary_color', 'unknown'),
                'material_type': parsed_response.get('material_composition', {}).get('likely_material', 'unknown'),
                'pattern': parsed_response.get('pattern_characteristics', {}).get('pattern_type', 'unknown'),
                'texture': parsed_response.get('texture_properties', {}).get('visual_texture', 'unknown'),
                'structural_elements': parsed_response.get('structural_elements', {}).get('details', [])
            }
            image_tags[os.path.basename(path)] = result
        except Exception as e:
            image_tags[os.path.basename(path)] = {"error": str(e)}
    return image_tags

def create_design_prompt(image_tags, control_image):
    model = genai.GenerativeModel('gemini-2.0-flash')
    fabric_descriptions = []
    for filename, tag_data in image_tags.items():
        if isinstance(tag_data, dict) and 'error' not in tag_data:
            colors = tag_data.get('color_palette', 'unknown')
            color_str = ', '.join(colors) if isinstance(colors, list) else colors
            desc = f"{filename}: {color_str} {tag_data['material_type']} with {tag_data['pattern']} pattern and {tag_data['texture']} texture"
            fabric_descriptions.append(desc)

    prompt = "\n".join([
        "You are a designer. This image is for structure reference only.",
        "Design a single, simple, realistic garment using the fabrics below:",
        "Materials:\n" + "\n".join(fabric_descriptions)
    ])
    response = model.generate_content([prompt, control_image], generation_config={"temperature": 0.2})
    return response.text.strip()

def generate_design(prompt, control_images, num_steps=20, guidance_scale=5):
    results = []
    for control_image in control_images:
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

def analyze_from_uploaded_files(file_objs):
    image_paths = save_uploaded_files(file_objs)
    analysis = analyze_images(image_paths)
    control_image = process_and_blend_images(image_paths)
    return json.dumps(analysis, indent=2), analysis, control_image

def final_design_wrapper(prompt, files):
    image_paths = save_uploaded_files(files)
    control_image = process_and_blend_images(image_paths)
    return generate_design(prompt, [control_image])[0]

# ========== GRADIO UI ==========

def load_images_from_files(files):
    return [Image.open(file) for file in files]

with gr.Blocks() as demo:
    gr.Markdown("## 🧥 Fabric Fusion Designer")

    with gr.Row():
        image_input = gr.File(file_types=["image"], label="Upload Fabric Images", file_count="multiple")
        analyze_button = gr.Button("Analyze Fabrics 🧠")

    uploaded_images_display = gr.Gallery(label="📸 Uploaded Fabric Images", columns=3, height="300px", object_fit="contain")
    image_input.change(fn=load_images_from_files, inputs=image_input, outputs=uploaded_images_display)

    analysis_output = gr.Textbox(label="📝 Fabric Analysis JSON", lines=15)
    control_image_display = gr.Image(label="🧭 Control Image", type="pil")
    prompt_state = gr.State()
    analysis_state = gr.State()
    control_image_state = gr.State()

    analyze_button.click(
        fn=analyze_from_uploaded_files,
        inputs=image_input,
        outputs=[analysis_output, analysis_state, control_image_display]
    ).then(
        lambda x: x, inputs=control_image_display, outputs=control_image_state
    )

    generate_prompt_button = gr.Button("Generate Prompt from Analysis 🧶")
    prompt_output = gr.Textbox(label="🧵 Generated Design Prompt", lines=4)

    generate_prompt_button.click(
        fn=create_design_prompt,
        inputs=[analysis_state, control_image_state],
        outputs=[prompt_output]
    ).then(
        lambda x: x,
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

demo.launch(share=False, show_api=False, prevent_thread_lock=True)
