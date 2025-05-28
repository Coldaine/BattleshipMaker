import google.generativeai as genai
import PIL.Image
import io
import os 
import logging 
import json 
import base64 
from src import config
from src.utils.cache import get_from_cache, set_to_cache

logger = logging.getLogger(__name__)

# Configure the Gemini API key
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
else:
    logger.info("GEMINI_API_KEY not found in config.py. Assuming it's set as an environment variable (e.g., GOOGLE_API_KEY).")

# Define the new, detailed JSON schema for the response
RESPONSE_SCHEMA = {
  "type": "object",
  "properties": {
    "view_type": {
      "type": "string",
      "description": "Either 'top_view', 'side_view', 'front_view', 'back_view', or 'perspective_view' of the ship in this image. Choose the most descriptive option.",
      "enum": ["top_view", "side_view", "front_view", "back_view", "perspective_view"]
    },
    "ship_identification": {
      "type": "string",
      "description": "The specific ship class or name, e.g., 'Iowa Class Battleship', 'Yamato'. If unknown, use 'Unknown Ship'. If only a generic type is identifiable (e.g., 'Destroyer', 'Cruiser'), provide that."
    },
    "transcribed_text": {
      "type": "string",
      "description": "Any discernible text from the image, such as hull numbers, names, or markings. If no text is visible, provide an empty string."
    },
    "hull": {
      "type": ["object", "null"], 
      "properties": {
        "description": {"type": "string", "description": "A textual description of the primary ship hull, excluding the superstructure. Emphasize its type (e.g., 'battleship hull')."},
        "bounding_box_2d": {"type": "array", "items": {"type": "integer"}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the hull, normalized to 0-1000 range. Omit if not identifiable."},
        "segmentation_mask_base64_png": {"type": "string", "description": "Base64 encoded PNG of the primary ship hull segmentation mask. This mask MUST precisely outline the main body of the ship, *explicitly excluding* the superstructure, masts, funnels, turrets, and any other distinct deck features. Omit this field if no clear, distinct, and separated hull is not identifiable."}
      }
    },
    "superstructure": {
      "type": ["object", "null"], 
      "properties": {
        "description": {"type": "string", "description": "A textual description of the ship\'s superstructure."},
        "bounding_box_2d": {"type": "array", "items": {"type": "integer"}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the superstructure, normalized to 0-1000 range. Omit if not identifiable."},
        "segmentation_mask_base64_png": {"type": "string", "description": "Base64 encoded PNG of the ship\'s superstructure segmentation mask. This mask should precisely outline all components typically considered part of the superstructure (e.g., bridge, command tower, masts, radar arrays, funnels, deck houses). Omit this field if no discernible superstructure is present."}
      }
    },
    "funnels": {
      "type": "array",
      "description": "An array of detected funnels on the ship. Each object includes a description, its bounding box, and its segmentation mask.",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string", "description": "Brief description of the funnel, e.g., 'forward funnel', 'aft funnel', 'twin funnel'"},
          "bounding_box_2d": {"type": "array", "items": {"type": "integer"}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the funnel, normalized to 0-1000 range. Omit if not identifiable."},
          "segmentation_mask_base64_png": {"type": "string", "description": "Base64 encoded PNG of the funnel\'s segmentation mask. Omit if not identifiable."}
        }
      },
      "default": []
    },
    "turrets": {
      "type": "array",
      "description": "An array of detected main gun turrets on the ship. Each object includes a description, its bounding box, and its segmentation mask.",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string", "description": "Brief description of the turret, e.g., 'forward main turret', 'aft superfiring turret', 'midships twin turret'"},
          "bounding_box_2d": {"type": "array", "items": {"type": "integer"}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the turret, normalized to 0-1000 range. Omit if not identifiable."},
          "segmentation_mask_base64_png": {"type": "string", "description": "Base64 encoded PNG of the turret\'s segmentation mask. Omit if not identifiable."}
        }
      },
      "default": []
    },
    "bridge_or_tower": {
      "type": ["object", "null"], 
      "description": "Details about the ship\'s primary bridge or command tower structure.",
      "properties": {
        "description": {"type": "string", "description": "Brief description of the bridge/tower, e.g., 'main bridge', 'conning tower bridge', 'superstructure tower'"},
        "bounding_box_2d": {"type": "array", "items": {"type": "integer"}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the bridge/tower, normalized to 0-1000 range. Omit if not identifiable."}
      }
    }
  },
  "required": ["view_type", "ship_identification", "transcribed_text"]
}

def analyze_image_with_gemini(image_pil, image_identifier):
    cache_key = f"gemini_analysis_v2_1.5flash_{image_identifier}" # Updated cache key for new model/prompt
    if not config.DEBUG_FLAG:
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Returning cached Gemini API response (v2 schema) for {image_identifier}")
            return cached_result

    logger.info(f"Calling Gemini API (v2 schema) for image: {image_identifier}")

    try:
        model = genai.GenerativeModel(
            config.GEMINI_API_MODEL,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        image_part = {
            "mime_type": "image/png",
            "data": img_bytes
        }

        # Constructing the schema string separately to avoid complex f-string issues with the tool
        # Using triple single quotes for the main f-string and double quotes for internal JSON strings.
        # Escaping backticks within the schema description for the prompt.
        schema_as_string = '''
{{
  "type": "object",
  "properties": {{
    "view_type": {{
      "type": "string",
      "description": "Either 'top_view', 'side_view', 'front_view', 'back_view', or 'perspective_view' of the ship in this image. Choose the most descriptive option.",
      "enum": ["top_view", "side_view", "front_view", "back_view", "perspective_view"]
    }},
    "ship_identification": {{
      "type": "string",
      "description": "The specific ship class or name, e.g., 'Iowa Class Battleship', 'Yamato'. If unknown, use 'Unknown Ship'. If only a generic type is identifiable (e.g., 'Destroyer', 'Cruiser'), provide that."
    }},
    "transcribed_text": {{
      "type": "string",
      "description": "Any discernible text from the image, such as hull numbers, names, or markings. If no text is visible, provide an empty string."
    }},
    "hull": {{
      "type": ["object", "null"],
      "properties": {{
        "description": {{"type": "string", "description": "A textual description of the primary ship hull, excluding the superstructure. Emphasize its type (e.g., 'battleship hull')."}},
        "bounding_box_2d": {{"type": "array", "items": {{"type": "integer"}}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the hull, normalized to 0-1000 range. Omit if not identifiable."}},
        "segmentation_mask_base64_png": {{"type": "string", "description": "Base64 encoded PNG of the primary ship hull's segmentation mask. This mask MUST precisely outline the main body of the ship, *explicitly excluding* the superstructure, masts, funnels, turrets, and any other distinct deck features. Omit this field if no clear, distinct, and separated hull is not identifiable."}}
      }}
    }},
    "superstructure": {{
      "type": ["object", "null"],
      "properties": {{
        "description": {{"type": "string", "description": "A textual description of the ship's superstructure."}},
        "bounding_box_2d": {{"type": "array", "items": {{"type": "integer"}}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the superstructure, normalized to 0-1000 range. Omit if not identifiable."}},
        "segmentation_mask_base64_png": {{"type": "string", "description": "Base64 encoded PNG of the ship's superstructure segmentation mask. This mask should precisely outline all components typically considered part of the superstructure (e.g., bridge, command tower, masts, radar arrays, funnels, deck houses). Omit this field if no discernible superstructure is present."}}
      }}
    }},
    "funnels": {{
      "type": "array",
      "description": "An array of detected funnels on the ship. Each object includes a description, its bounding box, and its segmentation mask.",
      "items": {{
        "type": "object",
        "properties": {{
          "description": {{"type": "string", "description": "Brief description of the funnel, e.g., 'forward funnel', 'aft funnel', 'twin funnel'"}},
          "bounding_box_2d": {{"type": "array", "items": {{"type": "integer"}}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the funnel, normalized to 0-1000 range. Omit if not identifiable."}},
          "segmentation_mask_base64_png": {{"type": "string", "description": "Base64 encoded PNG of the funnel's segmentation mask. Omit if not identifiable."}}
        }}
      }},
      "default": []
    }},
    "turrets": {{
      "type": "array",
      "description": "An array of detected main gun turrets on the ship. Each object includes a description, its bounding box, and its segmentation mask.",
      "items": {{
        "type": "object",
        "properties": {{
          "description": {{"type": "string", "description": "Brief description of the turret, e.g., 'forward main turret', 'aft superfiring turret', 'midships twin turret'"}},
          "bounding_box_2d": {{"type": "array", "items": {{"type": "integer"}}, "description": "2D bounding box coordinates [x_min, y_min, x_max, y_max] of the turret, normalized to 0-1000 range. Omit if not identifiable."}},
          "segmentation_mask_base64_png": {{"type": "string", "description": "Base64 encoded PNG of the turret's segmentation mask. Omit if not identifiable."}}
        }}
      }},
      "default": []
    }},
    "bridge_or_tower": {{
      "type": ["object", "null"],
      "description": "Details about the ship\'s primary bridge or command tower structure.",
      "properties": {{
        "description": {{"type": "string", "description": "Brief description of the bridge/tower, e.g., 'main bridge', 'conning tower bridge', 'superstructure tower'"}}
      }}
    }}
  }}
'''

        prompt_text = f'''Analyze this image of a ship.
My goal is to extract detailed information about its components for game asset customization.

Output a JSON object adhering to the following schema:
```json
{schema_as_string}
```
Ensure your entire response is a single JSON object matching this schema exactly.
For segmentation masks, provide them only if the object is clearly identifiable and the mask can be precise. If a mask is ambiguous or low quality, omit the \'segmentation_mask_base64_png\' field for that object.
'''

        response = model.generate_content([image_part, prompt_text])
        
        # Extract the JSON response
        # Updated to handle the new response structure
        # The 'response' object itself is the GenerateContentResponse
        if hasattr(response, 'candidates') and response.candidates and \
           hasattr(response.candidates[0], 'content') and response.candidates[0].content and \
           hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts and \
           hasattr(response.candidates[0].content.parts[0], 'text'):
            json_response = response.candidates[0].content.parts[0].text
        else:
            logger.error(f"Gemini API response did not contain expected path to text for {image_identifier}.")
            # Log the full response object for debugging
            logger.debug(f"Full Gemini API response object: {response!r}") # Use !r for a more detailed representation
            return None # Return None to indicate failure

        # Log the raw JSON response for debugging
        logger.info(f"Raw JSON response: {json_response}")

        # Parse the JSON response
        response_data = json.loads(json_response)

        # Cache the result if not in debug mode
        if not config.DEBUG_FLAG:
            set_to_cache(cache_key, response_data)

        return response_data

    except Exception as e:
        logger.error(f"Error analyzing image with Gemini: {e}")
        return None

if __name__ == '__main__':
    # Example usage (assuming an image file is provided)
    image_path = "path_to_your_ship_image.jpg"
    image_identifier = os.path.basename(image_path).split('.')[0]  # Use filename without extension as identifier

    # Open the image file
    with PIL.Image.open(image_path) as img:
        # Analyze the image and print the response
        analysis_result = analyze_image_with_gemini(img, image_identifier)
        print(json.dumps(analysis_result, indent=2))
