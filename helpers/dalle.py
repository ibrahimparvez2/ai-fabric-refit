import openai
import os
import uuid
import requests
import base64

def generate_dalle_image(prompt, output_dir):
    """Generate an image from a prompt using DALL·E 3 and save locally."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        # Generate an image using DALL·E 3 model
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        # Correct way to access the image URL
        image_url = response.data[0].url  # Access using dot notation, not dictionary

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(output_dir, f"{file_id}.png")

        # Download the image and save it locally
        img_data = requests.get(image_url).content
        with open(file_path, "wb") as handler:
            handler.write(img_data)

        # Return relative path to the saved image
        return f"/static/generated/{os.path.basename(file_path)}", None

    except openai.error.InvalidRequestError as e:
        # Handle invalid request (such as an inappropriate prompt)
        return "/static/placeholder-image.jpg", f"Inappropriate prompt: {e.user_message}"

    except openai.error.RateLimitError as e:
        # Handle rate limit exceeded errors
        return "/static/placeholder-image.jpg", "Rate limit exceeded. Please try again later."

    except Exception as e:
        # Catch any other general errors
        return "/static/placeholder-image.jpg", str(e)
