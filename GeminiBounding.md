In this experimental launch, we are providing developers with a powerful tool for object detection and localization within images and video. By accurately identifying and delineating objects with bounding boxes, developers can unlock a wide range of applications and enhance the intelligence of their projects.

Key Benefits:

Simple: Integrate object detection capabilities into your applications with ease, regardless of your computer vision expertise.
Customizable: Produce bounding boxes based on custom instructions (e.g. "I want to see bounding boxes of all the green objects in this image"), without having to train a custom model.
Technical Details:

Input: Your prompt and associated images or video frames.
Output: Bounding boxes in the [y_min, x_min, y_max, x_max] format. The top left corner is the origin. The x and y axis go horizontally and vertically, respectively. Coordinate values are normalized to 0-1000 for every image.
Visualization: AI Studio users will see bounding boxes plotted within the UI. Vertex AI users should visualize their bounding boxes through custom visualization code.
Gen AI SDK for Python
Install


pip install --upgrade google-genai
To learn more, see the SDK reference documentation.

Set environment variables to use the Gen AI SDK with Vertex AI:



# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True



import requests

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, Part, SafetySetting

from PIL import Image, ImageColor, ImageDraw

from pydantic import BaseModel

# Helper class to represent a bounding box
class BoundingBox(BaseModel):
    """
    Represents a bounding box with its 2D coordinates and associated label.

    Attributes:
        box_2d (list[int]): A list of integers representing the 2D coordinates of the bounding box,
                            typically in the format [x_min, y_min, x_max, y_max].
        label (str): A string representing the label or class associated with the object within the bounding box.
    """

    box_2d: list[int]
    label: str

# Helper function to plot bounding boxes on an image
def plot_bounding_boxes(image_uri: str, bounding_boxes: list[BoundingBox]) -> None:
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
        and their positions in normalized [y1 x1 y2 x2] format.
    """
    with Image.open(requests.get(image_uri, stream=True, timeout=10).raw) as im:
        width, height = im.size
        draw = ImageDraw.Draw(im)

        colors = list(ImageColor.colormap.keys())

        for i, bbox in enumerate(bounding_boxes):
            y1, x1, y2, x2 = bbox.box_2d
            abs_y1 = int(y1 / 1000 * height)
            abs_x1 = int(x1 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)

            color = colors[i % len(colors)]

            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
            )
            if bbox.label:
                draw.text((abs_x1 + 8, abs_y1 + 6), bbox.label, fill=color)

        im.show()

client = genai.Client(http_options=HttpOptions(api_version="v1"))

config = GenerateContentConfig(
    system_instruction="""
    Return bounding boxes as an array with labels.
    Never return masks. Limit to 25 objects.
    If an object is present multiple times, give each object a unique label
    according to its distinct characteristics (colors, size, position, etc..).
    """,
    temperature=0.5,
    safety_settings=[
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ],
    response_mime_type="application/json",
    response_schema=list[BoundingBox],  # Add BoundingBox class to the response schema
)

image_uri = "https://storage.googleapis.com/generativeai-downloads/images/socks.jpg"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents=[
        Part.from_uri(
            file_uri=image_uri,
            mime_type="image/jpeg",
        ),
        "Output the positions of the socks with a face. Label according to position in the image.",
    ],
    config=config,
)
print(response.text)
plot_bounding_boxes(image_uri, response.parsed)

# Example response:
# [
#     {"box_2d": [36, 246, 380, 492], "label": "top left sock with face"},
#     {"box_2d": [260, 663, 640, 917], "label": "top right sock with face"},
# ]


Briefing: Gemini 2.5 Vision AI - Image Segmentation and Masking Capabilities

Overview of Gemini Vision AI (General):
Gemini models are inherently multimodal, meaning they can understand and reason about various data types (text, images, audio, video) simultaneously. For visual input, they excel at tasks like object detection (bounding boxes), image captioning, and visual reasoning.

Key Advancement in Gemini 2.5 (from 1.5 Pro/Flash):

The most significant new capability in Gemini 2.5 (both Pro and Flash versions) related to your project is the direct generation of image segmentation masks in addition to 2D bounding boxes.

Segmentation Masks:

Definition: Unlike bounding boxes (which are rectangular), segmentation masks provide a pixel-level outline or contour of a specified object within an image. This allows for much more precise identification and separation of objects.
Output Format: Gemini 2.5 returns these segmentation masks as base64-encoded PNG images embedded within JSON strings. This means the AI agent will need to decode these strings and load them as images to interpret the mask.
Precision: While marked as "experimental," this capability can be remarkably precise and can isolate requested objects effectively.
Cost-Effectiveness: Using Gemini 2.5 Flash for segmentation is noted as being very inexpensive (fractions of a cent per image), making it viable for high-volume tasks.
Improved Bounding Boxes (Carry-over/Refinement):

Gemini models (including 2.5) continue to support generating 2D bounding boxes, often providing coordinates scaled between 0 and 100 or 0 and 1000, which then need to be scaled to the original image dimensions. This capability is now even more robust with 2.5's enhanced reasoning.
How Gemini 2.5's Vision Capabilities Benefit This Project:

More Precise Superstructure Identification (Pre-Cut):

Instead of just relying on bounding boxes for a rough Z-height estimate, a segmentation mask of the "superstructure" could provide a much more accurate and non-rectangular boundary. This mask, if accurately extracted and translated to 3D, could potentially guide a more complex cutting geometry in Blender than just a flat plane or simple box.
The model can reason about the semantic meaning of "superstructure" and identify it even if its shape is complex or non-standard across different ship models.
More Granular Verification (Post-Cut):

After the cutting operation, the AI can be prompted to check if the "superstructure" is completely gone. If any remnants are left, a segmentation mask would precisely highlight these remaining pixels, providing very specific feedback for debugging the cutting script or adjusting the cut_height. This is superior to just a "yes/no" answer, as it pinpoints the exact problematic areas.
Differences/Improvements Since Gemini 1.5:

Direct Segmentation Mask Output: This is the primary new feature for visual analysis. Gemini 1.5 Pro could reason about shapes and locations, but 2.5 models can directly generate the pixel-level masks.
Enhanced Reasoning and Context Window: Both 2.5 Pro and Flash have improved reasoning capabilities and large context windows (up to 1 million tokens, with plans for 2 million for Pro). This means they can process more complex visual information and handle more nuanced prompts for identification and verification.
Cost Efficiency (especially Flash): Gemini 2.5 Flash is specifically optimized for speed and cost, making the API calls for visual analysis (including segmentation) very affordable for batch processing.
Actionable Insight for the AI Agent:

When implementing the Gemini calls, specifically leverage the ability to request segmentation masks by crafting prompts that explicitly ask for them, often in a JSON format. This will provide the most detailed visual information for both identifying the superstructure and verifying its removal. The agent should be aware of the need to decode the base64 PNG mask data.

Example code: 

import google.generativeai as genai
from PIL import Image
import base64
import json
from io import BytesIO
import os

# --- Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual API key
# or load it securely, e.g., from an environment variable:
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # For quick testing, but use os.getenv() in production

genai.configure(api_key=GEMINI_API_KEY)

# Use a Gemini 2.5 model capable of vision tasks and segmentation.
# 'gemini-1.5-pro' is generally good; 'gemini-1.5-flash' is faster and cheaper.
# Note: As of my last update, specific "2.5" model names might include '-preview'
# or other suffixes. Always check the latest Google AI documentation for the exact model IDs.
# For segmentation, 'gemini-1.5-pro-latest' or 'gemini-1.5-flash-latest' should work.
model = genai.GenerativeModel('gemini-1.5-pro') # Using pro for robust reasoning

# --- Example Image (replace with your ship render) ---
# Create a dummy image for demonstration if you don't have one
try:
    img = Image.open("example_ship_render.png")
except FileNotFoundError:
    print("Creating a dummy image for demonstration...")
    # Create a simple red square image if no file is found
    img = Image.new('RGB', (600, 400), color = 'red')
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.text((10,10), "Dummy Ship Render", fill=(255,255,255))
    img.save("example_ship_render.png")
    print("Saved dummy image as example_ship_render.png")
    img = Image.open("example_ship_render.png")


# --- Prompting for Bounding Boxes and Segmentation Masks ---

# It's crucial to explicitly ask for the output format,
# especially for segmentation masks and bounding boxes.
# Requesting JSON output makes parsing much easier.
prompt_text = """
Analyze this image of a ship.
Identify the 'superstructure'.

Output a JSON object with the following structure:
{
  "superstructure": {
    "description": "A textual description of the identified superstructure.",
    "bounding_box_2d": [x_min, y_min, x_max, y_max], // Coordinates normalized to 0-1000
    "segmentation_mask_base64_png": "base64_encoded_png_data" // Base64 encoded PNG of the mask
  },
  "overall_ship_type": "e.g., Frigate, Cargo Ship, Battleship"
}
If no clear superstructure is found, omit 'bounding_box_2d' and 'segmentation_mask_base64_png' and set 'description' to 'None found'.
"""

try:
    print("Sending image and prompt to Gemini API...")
    response = model.generate_content([prompt_text, img])

    # Extract the JSON part from the response. Gemini often wraps JSON in markdown.
    response_text = response.text
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()

    data = json.loads(json_str)

    # --- Process Bounding Box (if present) ---
    superstructure_info = data.get('superstructure', {})
    bbox = superstructure_info.get('bounding_box_2d')
    description = superstructure_info.get('description')
    ship_type = data.get('overall_ship_type')

    print(f"\n--- Gemini Analysis ---")
    print(f"Description: {description}")
    print(f"Overall Ship Type: {ship_type}")
    if bbox:
        # Bounding box coordinates are typically normalized to 0-1000.
        # You'll need to scale them back to your original image dimensions.
        img_width, img_height = img.size
        x_min, y_min, x_max, y_max = bbox
        
        # Scale to original image pixels
        pixel_x_min = int(x_min / 1000 * img_width)
        pixel_y_min = int(y_min / 1000 * img_height)
        pixel_x_max = int(x_max / 1000 * img_width)
        pixel_y_max = int(y_max / 1000 * img_height)

        print(f"Bounding Box (normalized 0-1000): {bbox}")
        print(f"Bounding Box (pixel): [{pixel_x_min}, {pixel_y_min}, {pixel_x_max}, {pixel_y_max}]")

        # --- Process Segmentation Mask (if present) ---
        mask_base64 = superstructure_info.get('segmentation_mask_base64_png')
        if mask_base64:
            # Remove "data:image/png;base64," prefix if it exists
            if "base64," in mask_base64:
                mask_base64 = mask_base64.split("base64,")[1]

            mask_bytes = base64.b64decode(mask_base64)
            mask_image = Image.open(BytesIO(mask_bytes))

            print(f"Segmentation Mask received. Dimensions: {mask_image.size}")

            # You can now save or display the mask image.
            mask_image.save("superstructure_mask.png")
            print("Segmentation mask saved as superstructure_mask.png")

            # Optional: Overlay mask on original image for visualization
            # Ensure both images are in RGBA for proper alpha compositing
            original_rgba = img.convert("RGBA")
            # Convert mask to grayscale (L mode) or 1-bit (1 mode)
            # Then create an overlay from it
            mask_alpha = mask_image.convert("L") # L mode is grayscale 0-255
            # Create a colored overlay from the mask
            overlay_color = (255, 0, 0, 128) # Semi-transparent red
            overlay = Image.new("RGBA", mask_alpha.size, overlay_color)
            overlay.putalpha(mask_alpha) # Use the mask as the alpha channel

            # Resize overlay to match original image if needed (masks can be returned at different scales)
            if overlay.size != original_rgba.size:
                overlay = overlay.resize(original_rgba.size, Image.Resampling.LANCZOS)

            combined_image = Image.alpha_composite(original_rgba, overlay)
            combined_image.save("ship_with_superstructure_overlay.png")
            print("Combined image with mask overlay saved as ship_with_superstructure_overlay.png")
        else:
            print("No segmentation mask provided for superstructure.")
    else:
        print("No bounding box provided for superstructure (or superstructure not found).")

except json.JSONDecodeError as e:
    print(f"Error parsing JSON from Gemini response: {e}")
    print("Raw Gemini response:\n", response_text)
except Exception as e:
    print(f"An error occurred: {e}")