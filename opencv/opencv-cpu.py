# Based on gemini prompt how well would this work with openCV and pretrained diffusion models.. 
# TODO: this needs to run on GPU optimized hardware for now commented out relevant parts to work on machine


# Import required libraries
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import numpy as np
import cv2

# Set up GPU memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",  
    use_auth_token=True,
    # torch_dtype=torch.float16
).to(device)

# Initialize Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=True,
    controlnet=controlnet,
    low_cpu_mem_usage=True
    # torch_dtype=torch.float16
).to(device)

# Configure scheduler for better control
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()

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

def generate_design(prompt, control_images, num_steps=5, guidance_scale=7.5): # num steps reduced from 30 --> 5
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
        result = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        results.append(result)
    return results


if __name__ == "__main__":
    image_paths = [
        "images/jeans.jpg",
        "images/nike-hoodie.jpg",
        "images/shirt.jpg"
    ]

    # Create one blended control image
    control_image = process_and_blend_images(image_paths)
      # Gemini-flash-prompt needs to be shortened to under 80 tokens to work
    prompt = (
        "Create a lined denim jacket using three materials. Dark navy denim (jeans.jpg) forms the durable body." 
        "The nke-hoodie.jpg material makes a detachable hood with water resistance." 
        "Plaid cotton blend from shirt.jpg lines the jacket, adding warmth."
    )

    # Generate a single final design
    result = generate_design(prompt, [control_image])[0]

    result.save("final_design.jpg")
    result.show()