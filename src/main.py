import argparse
import os
import sys
import logging
import glob # For listing image files
from io import BytesIO # For decoding masks from Gemini
from PIL import Image # For opening decoded mask

# Add project root to sys.path to allow absolute imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    # sys.path.insert(0, PROJECT_ROOT) # Ensure this is handled if necessary
    pass

import numpy as np # Added numpy import

from src.utils.logger import setup_logger # Explicitly import and call

logger = setup_logger() # Move logger initialization here

from src.image_processing.loader import load_image_pil
from src.image_processing.splitter import split_image_if_needed
from src.utils.file_utils import get_filename_without_extension, sanitize_filename
from src import config
from src.gemini_api.client import analyze_image_with_gemini
from src.image_processing.vectorizer import vectorize_image_to_svg_subprocess, generate_hull_silhouette_mask
import base64

# Attempt to import trimesh, handle ImportError if not installed
try:
    import trimesh
    import trimesh.creation
    import trimesh.transformations as tf
    from trimesh.path import Path2D # Import Path2D specifically
    from trimesh.path.entities import Line # Import Line entity for Path2D
    TRIMESH_AVAILABLE = True
    logger.info("Trimesh library imported successfully.")
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("Trimesh library not found. 3D model generation will be skipped.")

# Attempt to import shapely, handle ImportError if not installed
try:
    from shapely.geometry import Polygon, LineString
    SHAPELY_AVAILABLE = True
    logger.info("Shapely library imported successfully.")
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely library not found. Advanced hull construction will be limited.")

# Helper function to get save path in flat type-based directories
def get_save_path_for_type(type_subdir_name, filename):
    base_output_dir = config.OUTPUT_DIR
    type_dir = os.path.join(base_output_dir, type_subdir_name)
    os.makedirs(type_dir, exist_ok=True)
    return os.path.join(type_dir, filename)

def process_single_image(image_path, args):
    logger.info(f"Processing image: {image_path}")
    image_filename = os.path.basename(image_path)
    # Use the sanitized filename (without extension) as the consistent ship_id for all views from this image.
    current_ship_id_for_image = sanitize_filename(get_filename_without_extension(image_filename))
    logger.info(f"Using '{current_ship_id_for_image}' as the primary ID for all views from {image_filename}")

    pil_image = load_image_pil(image_path)
    if pil_image is None:
        logger.error(f"Failed to load image {image_path}")
        return None, None # Return None for ship_views_data and ship_id

    # Split image if it contains multiple views (e.g., top and side)
    # This returns a list of dictionaries, each with 'view_pil_image' and 'view_type_guess'
    split_views = split_image_if_needed(pil_image)
    
    # This dictionary will store all data for the views from THIS specific image file
    # e.g., {"top_view": {...data...}, "side_view": {...data...}}
    ship_views_data_for_this_image = {}

    # Iterate directly over the list of PIL Images returned by split_image_if_needed
    for i, view_pil_image in enumerate(split_views):
        # Generate a view type guess based on the index
        # A more sophisticated approach could try to identify view type here based on image properties
        view_type_guess = f"view_{i+1}"
        
        # Use current_ship_id_for_image in filenames for clarity and consistency
        view_filename_base = f"{current_ship_id_for_image}_{view_type_guess}"

        original_view_save_path = get_save_path_for_type("images", f"{view_filename_base}_original_view.png")
        view_pil_image.save(original_view_save_path)
        logger.info(f"Saved original split view to: {original_view_save_path}")

        # Analyze the individual view with Gemini
        # Pass current_ship_id_for_image as the ship_id argument to Gemini client
        # The view_type_guess and debug flag are handled within the analyze_image_with_gemini function.
        analysis_result = analyze_image_with_gemini(view_pil_image, current_ship_id_for_image)

        if analysis_result:
            # Extract the actual view_type identified by Gemini, if available.
            # If Gemini doesn't provide one, we stick with our view_type_guess (e.g., 'view_1', 'view_2').
            actual_view_type = analysis_result.get("view_type", view_type_guess).lower()
            if not actual_view_type:
                 actual_view_type = f"unknown_view_{i+1}"
                 logger.warning(f"Gemini did not return a view_type for {view_filename_base}, using '{actual_view_type}'.")
            
            # Store the entire analysis_result under the actual_view_type for the current_ship_id_for_image.
            ship_views_data_for_this_image[actual_view_type] = analysis_result
            logger.info(f"Stored Gemini analysis for '{current_ship_id_for_image}' - view '{actual_view_type}'")

            # --- Output Gemini analysis step results to folder ---
            import json
            analysis_output_dir = get_save_path_for_type("analysis", "dummy.txt")  # get path, then strip filename
            analysis_output_dir = os.path.dirname(analysis_output_dir)
            os.makedirs(analysis_output_dir, exist_ok=True)
            # Save full Gemini analysis result as JSON
            analysis_json_path = os.path.join(analysis_output_dir, f"{view_filename_base}_gemini_analysis.json")
            with open(analysis_json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved Gemini analysis JSON to: {analysis_json_path}")
            # Optionally, save bounding boxes/components as a separate JSON for quick inspection
            bboxes = {}
            for comp in ["hull", "superstructure", "turrets", "funnels", "bridge_or_tower"]:
                if comp in analysis_result:
                    bboxes[comp] = analysis_result[comp]
            bboxes_json_path = os.path.join(analysis_output_dir, f"{view_filename_base}_bboxes.json")
            with open(bboxes_json_path, 'w', encoding='utf-8') as f:
                json.dump(bboxes, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved bounding boxes/components JSON to: {bboxes_json_path}")

            # --- Visualization: Draw bounding boxes on the original view image ---
            from PIL import ImageDraw, ImageFont
            vis_img = view_pil_image.convert('RGB').copy()
            draw = ImageDraw.Draw(vis_img)
            color_map = {
                'hull': 'red',
                'superstructure': 'blue',
                'turrets': 'green',
                'funnels': 'orange',
                'bridge_or_tower': 'purple'
            }
            # Draw single-instance components
            for comp in ['hull', 'superstructure', 'bridge_or_tower']:
                comp_data = analysis_result.get(comp)
                if comp_data and isinstance(comp_data, dict) and 'bounding_box_2d' in comp_data:
                    bbox = comp_data['bounding_box_2d']
                    draw.rectangle(bbox, outline=color_map[comp], width=3)
                    draw.text((bbox[0], bbox[1]), comp, fill=color_map[comp])
            # Draw multi-instance components
            for comp in ['turrets', 'funnels']:
                comp_list = analysis_result.get(comp, [])
                if isinstance(comp_list, list):
                    for idx, item in enumerate(comp_list):
                        if isinstance(item, dict) and 'bounding_box_2d' in item:
                            bbox = item['bounding_box_2d']
                            label = f"{comp}_{idx+1}"
                            draw.rectangle(bbox, outline=color_map[comp], width=3)
                            draw.text((bbox[0], bbox[1]), label, fill=color_map[comp])
            vis_path = os.path.join(analysis_output_dir, f"{view_filename_base}_bboxes_overlay.png")
            vis_img.save(vis_path)
            logger.info(f"Saved bounding box overlay visualization to: {vis_path}")

            transcribed_text = analysis_result.get("transcribed_text", "")
            if transcribed_text:
                transcribed_text_save_path = get_save_path_for_type("text", f"{view_filename_base}_transcribed.txt")
                with open(transcribed_text_save_path, 'w', encoding='utf-8') as f:
                    f.write(transcribed_text)
                logger.info(f"Saved transcribed text to: {transcribed_text_save_path}")

            hull_silhouette_mask_base64 = analysis_result.get("hull_silhouette_mask_base64")
            hull_silhouette_mask_pil = None
            if hull_silhouette_mask_base64:
                try:
                    mask_bytes = base64.b64decode(hull_silhouette_mask_base64)
                    hull_silhouette_mask_pil = Image.open(BytesIO(mask_bytes)).convert('L')
                    mask_save_path = get_save_path_for_type("images", f"{view_filename_base}_debug_gemini_hull_mask.png")
                    hull_silhouette_mask_pil.save(mask_save_path)
                    logger.info(f"Saved Gemini-provided hull mask to: {mask_save_path}")
                except Exception as e:
                    logger.error(f"Error decoding or saving Gemini hull mask for {view_filename_base}: {e}")
                    hull_silhouette_mask_pil = None
            
            if hull_silhouette_mask_pil is None and config.USE_OPENCV_FALLBACK_FOR_HULL_MASK:
                logger.info(f"Generating hull mask with OpenCV for {view_filename_base}...")
                try:
                    # Pass the actual_view_type to generate_hull_silhouette_mask
                    hull_silhouette_mask_pil = generate_hull_silhouette_mask(view_pil_image, actual_view_type)
                    if hull_silhouette_mask_pil:
                        mask_save_path = get_save_path_for_type("images", f"{view_filename_base}_debug_opencv_hull_mask.png")
                        hull_silhouette_mask_pil.save(mask_save_path)
                        logger.info(f"Saved OpenCV-generated hull mask to: {mask_save_path}")
                    else:
                        logger.warning(f"OpenCV hull mask generation failed for {view_filename_base}.")
                except Exception as e:
                    logger.error(f"Error generating OpenCV hull mask for {view_filename_base}: {e}", exc_info=True)
                    hull_silhouette_mask_pil = None

            # Store the PIL mask (either from Gemini or OpenCV) in a structured way for later use
            # This needs to be part of the analysis_result for the specific view
            if actual_view_type not in ship_views_data_for_this_image: # Should be redundant due to earlier assignment
                ship_views_data_for_this_image[actual_view_type] = {}
            if "processing_data" not in ship_views_data_for_this_image[actual_view_type]:
                ship_views_data_for_this_image[actual_view_type]["processing_data"] = {}
            ship_views_data_for_this_image[actual_view_type]["processing_data"]["hull_silhouette_mask_pil"] = hull_silhouette_mask_pil

            svg_output_path_best = get_save_path_for_type("vector", f"{view_filename_base}_best_detail.svg")
            vectorize_image_to_svg_subprocess(view_pil_image, svg_output_path_best)
            logger.info(f"Vectorized (best detail) view to: {svg_output_path_best}")

            if hull_silhouette_mask_pil and config.VECTORIZE_HULL_MASK_FALLBACK:
                svg_output_path_hull_fallback = get_save_path_for_type("vector", f"opencv_fallback_{view_filename_base}_hull_shape_only.svg")
                vectorize_image_to_svg_subprocess(hull_silhouette_mask_pil, svg_output_path_hull_fallback)
                logger.info(f"Vectorized (OpenCV fallback hull mask) to: {svg_output_path_hull_fallback}")
        else:
            logger.warning(f"No analysis result from Gemini for {view_filename_base}. Skipping this view.")
            
    return ship_views_data_for_this_image, current_ship_id_for_image

def apply_3d_framework_steps_for_ship(ship_id, views_data):
    logger.info(f"Starting 3D framework steps for {ship_id}")

    if not TRIMESH_AVAILABLE:
        logger.error(f"Skipping 3D framework steps for {ship_id}: Trimesh library is not available.")
        return

    # --- Step 3: Scaling & Alignment ---
    logger.info("Step 3: Scaling & Alignment")
    scale_factors = {"top_view": None, "side_view": None}
    actual_length_meters = None

    # Try to get actual length from transcribed dimensions in either view
    for view_type, view_data in views_data.items():
        if view_data and "dimensions_transcribed" in view_data:
            for dim in view_data.get("dimensions_transcribed", []):
                if dim.get("label") == "overall_length_L":
                    value_str = dim.get("value", "")
                    try:
                        # Basic parsing (needs improvement for robustness)
                        if "m" in value_str:
                            actual_length_meters = float(value_str.replace("m", "").strip())
                            logger.info(f"Found overall_length_L: {actual_length_meters}m from {view_type}")
                            break # Found the length, no need to check other dimensions
                    except ValueError:
                        logger.warning(f"Could not parse overall_length_L value: {value_str}")
            if actual_length_meters is not None:
                break # Found the length, exit outer loop

    if actual_length_meters is not None:
        # Calculate pixel length of hull from bounding box in side view
        side_view_data = views_data.get("side_view")
        if side_view_data:
            hull_bbox_side = side_view_data.get("hull", {}).get("bounding_box_2d")
            if hull_bbox_side and len(hull_bbox_side) == 4:
                pixel_length_side = hull_bbox_side[2] - hull_bbox_side[0] # x2 - x1
                if pixel_length_side > 0:
                    scale_factors["side_view"] = pixel_length_side / actual_length_meters
                    logger.info(f"Derived side view scale factor: {scale_factors['side_view']} pixels/meter")
                else:
                     logger.warning("Side view hull bounding box has zero or negative width.")
            else:
                logger.warning("Side view hull bounding box not found or invalid for scale calculation.")

        # For top view, we might need a width/beam dimension or assume aspect ratio
        # For simplicity in this step, let's assume the same scale factor or a default if width is not available.
        # A more robust implementation would look for a width dimension and its bbox in the top view.
        # If no top view specific scale can be derived, use the side view scale as a fallback or a default.
        scale_factors["top_view"] = scale_factors["side_view"] if scale_factors["side_view"] is not None else 5.0 # Example default

    else:
        logger.warning("Overall length dimension not found. Using default scale factors.")
        scale_factors = {"top_view": 5.0, "side_view": 5.0} # Example default scale factors

    logger.info(f"Final scale factors for {ship_id}: Top={scale_factors['top_view']}, Side={scale_factors['side_view']}")

    # --- Step 4: Hull Contour Extraction & Vectorization ---
    logger.info("Step 4: Hull Contour Extraction & Vectorization")
    
    hull_contours_meters = {}

    for view_type, view_data in views_data.items():
        hull_mask_pil = view_data.get("processing_data", {}).get("hull_silhouette_mask_pil")
        scale_factor = scale_factors.get(view_type)

        if hull_mask_pil and scale_factor is not None:
            logger.info(f"Extracting and converting hull contour for {view_type} to meters.")
            try:
                # Convert PIL Image (mask) to OpenCV format (NumPy array)
                # Ensure the mask is a binary image (0 or 255)
                hull_mask_np = np.array(hull_mask_pil.convert('L')) # Convert to grayscale if not already
                
                # Find contours. RETR_EXTERNAL retrieves only the outer contours.
                # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
                # and leaves only their end points.
                # Note: findContours modifies the input image, so we might work on a copy if needed later.
                # For now, it's the last use of this mask image data in this function.
                
                # OpenCV findContours requires image to be CV_8UC1, which is equivalent to uint8
                # The mask is already converted to 'L' mode (8-bit pixels, grayscale) and then to numpy array (uint8)
                
                # Add a check for OpenCV availability and import it here if not already imported globally
                try:
                    import cv2
                except ImportError:
                    logger.error("OpenCV is not installed. Cannot perform contour extraction.")
                    hull_contours_meters[view_type] = None
                    continue

                # Find contours - cv2.findContours returns a tuple depending on OpenCV version
                contours_data = cv2.findContours(hull_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

                if contours:
                    # Assume the largest contour is the main hull outline
                    main_contour_px = max(contours, key=cv2.contourArea)

                    # Simplify contour (optional but good for reducing vertices)
                    epsilon = 0.001 * cv2.arcLength(main_contour_px, True) # Adjust epsilon as needed
                    simplified_contour_px = cv2.approxPolyDP(main_contour_px, epsilon, True)

                    # Convert pixel coordinates to real-world coordinates (meters)
                    simplified_contour_meters = []
                    # The contour points are in [x, y] format relative to the mask image.
                    # We need to consider the original image's coordinate system if the mask was a cropped section.
                    # However, the current mask generation seems to be for the full split view image.
                    # So, we can directly apply the scale factor.
                    
                    # The y-axis in image coordinates typically points downwards.
                    # In a 3D world coordinate system (e.g., Unity), Y is typically up, and Z is forward/depth.
                    # For a side view (X-Y plane in 3D), image X maps to world X, image Y maps to world Y (with inversion).
                    # For a top view (X-Z plane in 3D), image X maps to world X, image Y maps to world Z.
                    
                    # Let's assume for side view: world X = image X, world Y = -image Y (or adjust origin)
                    # Let's assume for top view: world X = image X, world Z = image Y
                    
                    # This mapping needs careful consideration based on the desired 3D coordinate system.
                    # For now, a simple conversion assuming image origin (0,0) is a reference point:
                    
                    if view_type == "side_view": # Mapping to X-Y plane
                        for point_px in simplified_contour_px:
                            px, py = point_px[0]
                            real_x = px / scale_factor
                            real_y = -py / scale_factor # Invert Y for typical 3D coordinate systems
                            simplified_contour_meters.append((real_x, real_y))
                    elif view_type == "top_view": # Mapping to X-Z plane
                         for point_px in simplified_contour_px:
                            px, py = point_px[0]
                            real_x = px / scale_factor
                            real_z = py / scale_factor
                            simplified_contour_meters.append((real_x, real_z))
                    else:
                         logger.warning(f"Unknown view type {view_type} for contour conversion.")
                         simplified_contour_meters = None

                    hull_contours_meters[view_type] = simplified_contour_meters
                    logger.info(f"Extracted {len(simplified_contour_meters)} points for {view_type} hull contour in meters.")
                else:
                    logger.warning(f"No contours found in hull mask for {view_type}.")
                    hull_contours_meters[view_type] = None

            except Exception as e:
                logger.error(f"Error extracting/converting hull contour for {view_type}: {e}", exc_info=True)
                hull_contours_meters[view_type] = None
        else:
            logger.warning(f"Hull mask or scale factor not available for {view_type}. Cannot extract contour in meters.")
            hull_contours_meters[view_type] = None

    # Store the contours in meters in the views_data for later steps
    for view_type, contour_data in hull_contours_meters.items():
         if contour_data is not None:
             if "processing_data" not in views_data[view_type]:
                 views_data[view_type]["processing_data"] = {}
             views_data[view_type]["processing_data"]["hull_contour_meters"] = contour_data

    # --- Step 4.5: Generate 3D parameters for components (NEW) ---
    def compute_component_3d_params(views_data, scale_factors):
        """
        For each component (superstructure, turrets, funnels, bridge_or_tower),
        compute 3D bounding box center and extents in meters using top and side view bounding boxes.
        Store results in views_data["generated_components_3d_params"].
        """
        comp_types = ["superstructure", "funnels", "turrets", "bridge_or_tower"]
        params = {}
        for comp_type in comp_types:
            # Handle list components (turrets, funnels) and singletons
            for idx in range(10): # Arbitrary max count
                comp_name = f"{comp_type}_{idx+1}" if comp_type in ["turrets", "funnels"] else comp_type
                # Get bbox from top and side views
                top_view = views_data.get("top_view", {})
                side_view = views_data.get("side_view", {})
                if comp_type in ["turrets", "funnels"]:
                    top_comp = (top_view.get(comp_type) or [])
                    side_comp = (side_view.get(comp_type) or [])
                    if idx >= len(top_comp) and idx >= len(side_comp):
                        break # No more components
                    bbox_top = top_comp[idx]["bounding_box_2d"] if idx < len(top_comp) and "bounding_box_2d" in top_comp[idx] else None
                    bbox_side = side_comp[idx]["bounding_box_2d"] if idx < len(side_comp) and "bounding_box_2d" in side_comp[idx] else None
                else:
                    comp_top = top_view.get(comp_type)
                    comp_side = side_view.get(comp_type)
                    bbox_top = comp_top.get("bounding_box_2d") if comp_top and isinstance(comp_top, dict) else None
                    bbox_side = comp_side.get("bounding_box_2d") if comp_side and isinstance(comp_side, dict) else None
                    if not bbox_top and not bbox_side:
                        continue
                # Compute center and extents
                if bbox_top and bbox_side and scale_factors["top_view"] and scale_factors["side_view"]:
                    # bbox: [x1, y1, x2, y2] (pixels)
                    x1t, y1t, x2t, y2t = bbox_top
                    x1s, y1s, x2s, y2s = bbox_side
                    # In top view: x = X, y = Z; in side view: x = X, y = Y
                    center_x = ((x1t + x2t) / 2) / scale_factors["top_view"]
                    center_y = ((y1s + y2s) / 2) / scale_factors["side_view"]
                    center_z = ((y1t + y2t) / 2) / scale_factors["top_view"]
                    dim_x = abs(x2t - x1t) / scale_factors["top_view"]
                    dim_y = abs(y2s - y1s) / scale_factors["side_view"]
                    dim_z = abs(y2t - y1t) / scale_factors["top_view"]
                    params[comp_name] = {
                        "center_m": [center_x, center_y, center_z],
                        "dimensions_m": [dim_x, dim_y, dim_z]
                    }
                else:
                    # Not enough data for this component
                    continue
        return params

    # Compute and store 3D params for components
    views_data["generated_components_3d_params"] = compute_component_3d_params(views_data, scale_factors)

    # --- Step 5: Component Primitive Generation ---
    logger.info("Step 5: Component Primitive Generation")
    
    generated_components_3d = {}
    # List of component types we expect Gemini to identify and for which we'll try to generate primitives
    component_types_to_process = ["superstructure", "funnels", "turrets", "bridge_or_tower"]

    for comp_type in component_types_to_process:
        top_view_data = views_data.get("top_view")
        side_view_data = views_data.get("side_view")

        # Gemini's response structure has lists for 'funnels' and 'turrets', and objects for others.
        # We need to handle this difference.
        if comp_type in ["funnels", "turrets"]:
            top_view_components = top_view_data.get(comp_type, []) if top_view_data else []
            side_view_components = side_view_data.get(comp_type, []) if side_view_data else []

            # Iterate through instances of these components (e.g., multiple turrets)
            # This assumes a correspondence between the order/number of components in top and side views,
            # which might not always be accurate. A more robust approach would involve matching components.
            # For simplicity here, we'll pair them by index.
            num_components = max(len(top_view_components), len(side_view_components))

            for i in range(num_components):
                top_comp = top_view_components[i] if i < len(top_view_components) else None
                side_comp = side_view_components[i] if i < len(side_view_components) else None

                comp_name = f"{comp_type}_{i+1}"

                # Retrieve calculated 3D parameters
                comp_3d_params = views_data.get("generated_components_3d_params", {}).get(comp_name)

                if comp_3d_params and "dimensions_m" in comp_3d_params and "center_m" in comp_3d_params:
                    dims_m = comp_3d_params["dimensions_m"]
                    center_m = comp_3d_params["center_m"]
                    logger.debug(f"Generating 3D primitive for {comp_name}: Dims={dims_m}, Center={center_m}")

                    try:
                        # Create a box primitive using trimesh
                        # trimesh.creation.box creates a box centered at the origin with specified extents.
                        # We need to translate it to the calculated center_m.
                        primitive_mesh = trimesh.creation.box(extents=dims_m)

                        # Calculate the translation vector to move the box from origin to center_m
                        # The box is initially centered at [0,0,0]. We want its center to be at center_m.
                        # So, the translation is simply center_m.
                        translation_vector = np.array(center_m)
                        primitive_mesh.apply_translation(translation_vector)

                        generated_components_3d[comp_name] = primitive_mesh
                        logger.info(f"Generated 3D primitive (box) for {comp_name}.")

                    except Exception as e:
                        logger.error(f"Error generating 3D primitive for {comp_name}: {e}", exc_info=True)
                        generated_components_3d[comp_name] = {"error": "3D primitive generation failed"}
                else:
                    logger.warning(f"Missing 3D parameters for {comp_name}. Cannot generate primitive.")

        else: # Handle single object components like superstructure, bridge_or_tower
            comp_name = comp_type # Use comp_type as name for single instances
            top_comp = top_view_data.get(comp_type) if top_view_data else None
            side_comp = side_view_data.get(comp_type) if side_view_data else None

            # Retrieve calculated 3D parameters
            comp_3d_params = views_data.get("generated_components_3d_params", {}).get(comp_name)

            if comp_3d_params and "dimensions_m" in comp_3d_params and "center_m" in comp_3d_params:
                dims_m = comp_3d_params["dimensions_m"]
                center_m = comp_3d_params["center_m"]
                logger.debug(f"Generating 3D primitive for {comp_name}: Dims={dims_m}, Center={center_m}")

                try:
                    # Create a box primitive using trimesh
                    primitive_mesh = trimesh.creation.box(extents=dims_m)
                    translation_vector = np.array(center_m)
                    primitive_mesh.apply_translation(translation_vector)

                    generated_components_3d[comp_name] = primitive_mesh
                    logger.info(f"Generated 3D primitive (box) for {comp_name}.")

                except Exception as e:
                    logger.error(f"Error generating 3D primitive for {comp_name}: {e}", exc_info=True)
                    generated_components_3d[comp_name] = {"error": "3D primitive generation failed"}
            else:
                logger.warning(f"Missing 3D parameters for {comp_name}. Cannot generate primitive.")

    # Store the generated 3D component meshes
    views_data["generated_components_3d"] = generated_components_3d

    # --- Step 6: Hull Construction ---
    logger.info("Step 6: Hull Construction")
    hull_3d_mesh = None
    top_contour_m = views_data.get("top_view", {}).get("processing_data", {}).get("hull_contour_meters")
    side_contour_m = views_data.get("side_view", {}).get("processing_data", {}).get("hull_contour_meters")

    if top_contour_m and side_contour_m and SHAPELY_AVAILABLE:
        logger.info("Attempting to construct 3D hull from contours using Shapely and Trimesh.")
        try:
            min_x = min(min(p[0] for p in top_contour_m), min(p[0] for p in side_contour_m))
            max_x = max(max(p[0] for p in top_contour_m), max(p[0] for p in side_contour_m))
            num_cross_sections = 20
            x_sections = np.linspace(min_x, max_x, num_cross_sections)
            hull_cross_sections_paths = [] # Store Path2D objects directly
            
            top_poly = Polygon(top_contour_m) if len(top_contour_m) >= 3 else None
            side_poly = Polygon(side_contour_m) if len(side_contour_m) >= 3 else None

            if top_poly and side_poly:
                min_y_overall = min(p[1] for p in side_contour_m)
                max_y_overall = max(p[1] for p in side_contour_m)
                min_z_overall = min(p[1] for p in top_contour_m) # Top contour Y is world Z
                max_z_overall = max(p[1] for p in top_contour_m)

                # Generate a cross-section (Y-Z profile) at each X-coordinate
                for x_coord in x_sections:
                    # Create a vertical line segment at this X-coordinate spanning the overall Y and Z range
                    # For side view (XY plane), the vertical line is in Y.
                    # For top view (XZ plane), the vertical line is in Z.
                    
                    # Side view intersection line (in XY plane, constant X)
                    side_intersection_line = LineString([(x_coord, min_y_overall - 1), (x_coord, max_y_overall + 1)]) # Extend slightly beyond range
                    
                    # Top view intersection line (in XZ plane, constant X)
                    top_intersection_line = LineString([(x_coord, min_z_overall - 1), (x_coord, max_z_overall + 1)]) # Extend slightly beyond range

                    # Find intersection points
                    side_intersections = side_poly.exterior.intersection(side_intersection_line)
                    top_intersections = top_poly.exterior.intersection(top_intersection_line)

                    # Extract Y values from side intersections and Z values from top intersections
                    def extract_sorted_values(intersections):
                        values = []
                        if not intersections.is_empty:
                            if intersections.geom_type == 'Point':
                                values.append(intersections.y)
                            elif intersections.geom_type == 'MultiPoint':
                                values.extend([p.y for p in intersections.geoms])
                            elif intersections.geom_type == 'LineString':
                                values.extend([intersections.bounds[1], intersections.bounds[3]])
                        return sorted(list(set(values)))

                    side_y_values = extract_sorted_values(side_intersections)
                    top_z_values = extract_sorted_values(top_intersections)

                    # Only use cross-sections where both side and top have exactly two intersection points
                    if len(side_y_values) == 2 and len(top_z_values) == 2:
                        min_y_at_x, max_y_at_x = side_y_values
                        min_z_at_x, max_z_at_x = top_z_values
                        profile_points_yz = [
                            (min_y_at_x, min_z_at_x), (max_y_at_x, min_z_at_x),
                            (max_y_at_x, max_z_at_x), (min_y_at_x, max_z_at_x),
                            (min_y_at_x, min_z_at_x)
                        ]
                        try:
                            profile_path = Path2D(vertices=np.array(profile_points_yz))
                            transform_matrix = tf.translation_matrix([x_coord, 0, 0])
                            profile_path.apply_transform(transform_matrix)
                            hull_cross_sections_paths.append(profile_path)
                            logger.debug(f"Generated and transformed cross-section Path2D at X={x_coord}m.")
                        except Exception as path_e:
                            logger.error(f"Error creating/transforming trimesh.Path2D at X={x_coord}m: {path_e}")
                    else:
                        logger.warning(f"Skipping cross-section at X={x_coord}m: side_y_values={side_y_values}, top_z_values={top_z_values}")
            else:
                logger.warning("Top or side contour Polygon could not be created for cross-sections.")

            if hull_cross_sections_paths:
                logger.info(f"Generated {len(hull_cross_sections_paths)} cross-sections. Attempting lofting with Trimesh skin_sections.")
                # The sections are already transformed to their X-positions.
                hull_3d_mesh = trimesh.creation.skin_sections(sections=hull_cross_sections_paths, repair=True)
                if hull_3d_mesh and hull_3d_mesh.is_watertight:
                    logger.info("Hull lofting with Trimesh skin_sections successful and mesh is watertight.")
                elif hull_3d_mesh:
                    logger.warning("Hull lofting with Trimesh skin_sections completed, but mesh may not be watertight or valid.")
                else:
                    logger.error("Trimesh skin_sections failed to produce a mesh.")
            else:
                logger.warning("No valid cross-sections generated for hull lofting.")

        except Exception as loft_e:
            logger.error(f"Error during detailed hull construction: {loft_e}", exc_info=True)
            hull_3d_mesh = None
    
    if hull_3d_mesh is None: # Fallback to placeholder box if detailed construction fails or not possible
        logger.warning("Using placeholder box for hull as detailed construction failed or was skipped.")
        # Simplified placeholder box logic (ensure min_x, max_x etc. are defined if this path is taken)
        # This requires overall dimensions to be estimated if not from contours.
        # For now, let's assume if contours were not available, this step is largely skipped.
        # If contours were available but lofting failed, we can use their extents.
        if top_contour_m and side_contour_m: # Check if contours were available for extents
            all_x = [p[0] for p in top_contour_m] + [p[0] for p in side_contour_m]
            all_y_side = [p[1] for p in side_contour_m]
            all_z_top = [p[1] for p in top_contour_m] # top contour y is world z

            hull_center_x = (min(all_x) + max(all_x)) / 2
            hull_length = max(all_x) - min(all_x)
            hull_center_y = (min(all_y_side) + max(all_y_side)) / 2
            hull_height = max(all_y_side) - min(all_y_side)
            hull_center_z = (min(all_z_top) + max(all_z_top)) / 2
            hull_width = max(all_z_top) - min(all_z_top)

            hull_extents = [hull_length, hull_height, hull_width] # X, Y, Z extents
            hull_center = [hull_center_x, hull_center_y, hull_center_z]
            
            if any(e <= 0 for e in hull_extents):
                logger.error(f"Cannot create placeholder hull box with non-positive extents: {hull_extents}")
                hull_3d_mesh = None
            else:
                hull_3d_mesh = trimesh.creation.box(extents=hull_extents, transform=tf.translation_matrix(hull_center))
                logger.info(f"Created placeholder hull box with extents {hull_extents} centered at {hull_center}.")
        else:
            logger.error("Cannot create placeholder hull: contour data for extents not available.")
            hull_3d_mesh = None

    views_data["hull_3d_mesh"] = hull_3d_mesh

    # --- Step 8: Assembly & Relative Positioning ---
    logger.info("Step 8: Assembly & Relative Positioning")
    all_meshes_to_assemble = []
    if views_data.get("hull_3d_mesh"):
        all_meshes_to_assemble.append(views_data["hull_3d_mesh"])
    
    for comp_name, comp_data in views_data.get("generated_components_3d", {}).items():
        if isinstance(comp_data, trimesh.Trimesh):
            all_meshes_to_assemble.append(comp_data)
        elif isinstance(comp_data, dict) and "error" in comp_data:
            logger.warning(f"Skipping assembly of {comp_name} due to previous error: {comp_data['error']}")
        else:
            logger.warning(f"Skipping assembly of {comp_name}: Not a valid Trimesh object.")

    assembled_ship_mesh = None
    if all_meshes_to_assemble:
        if len(all_meshes_to_assemble) > 1:
            try:
                assembled_ship_mesh = trimesh.util.concatenate(all_meshes_to_assemble)
                logger.info(f"Assembled {len(all_meshes_to_assemble)} meshes into a single ship model.")
            except Exception as e:
                logger.error(f"Error during mesh assembly: {e}", exc_info=True)
                # Fallback: use hull if available, or the first component if no hull
                assembled_ship_mesh = all_meshes_to_assemble[0]
                logger.warning("Assembly failed. Using the first available mesh as the model.")
        else:
            assembled_ship_mesh = all_meshes_to_assemble[0]
            logger.info("Only one mesh part available. Using it as the assembled model.")
    else:
        logger.warning(f"No 3D meshes available to assemble for {ship_id}.")

    # --- Step 10: Export ---
    if assembled_ship_mesh:
        logger.info("Step 10: Exporting assembled 3D model")
        export_filename = f"{sanitize_filename(ship_id)}_model.obj"
        export_path = get_save_path_for_type("3d_models", export_filename) # New subdir for 3D models
        try:
            assembled_ship_mesh.export(export_path)
            logger.info(f"Exported assembled 3D model for {ship_id} to: {export_path}")
        except Exception as e:
            logger.error(f"Error exporting 3D model for {ship_id}: {e}", exc_info=True)
    else:
        logger.warning(f"No assembled 3D model to export for {ship_id}.")

def main(args):
    logger.info("Starting main processing pipeline.")
    
    all_ships_data = {}

    if args.image_path:
        image_files = [args.image_path]
    elif args.sample_dir:
        image_files = glob.glob(os.path.join(args.sample_dir, '*.jpg')) + \
                      glob.glob(os.path.join(args.sample_dir, '*.png'))
        if args.num_images > 0:
            image_files = image_files[:args.num_images]
    else:
        logger.error("No image path or sample directory provided. Exiting.")
        return

    if not image_files:
        logger.warning("No image files found to process.")
        return

    logger.info(f"Found {len(image_files)} image(s) to process: {image_files}")

    for image_path in image_files:
        ship_views_data_for_current_image, ship_id_for_current_image = process_single_image(image_path, args)
        
        if ship_views_data_for_current_image and ship_id_for_current_image:
            all_ships_data[ship_id_for_current_image] = ship_views_data_for_current_image
            logger.info(f"Aggregated all views for ship ID '{ship_id_for_current_image}' from image {image_path}")
        else:
            logger.warning(f"No view data returned from process_single_image for {image_path}. Skipping aggregation for this image.")

    logger.info("\n--- Starting 3D Model Generation Phase ---")
    for ship_id, views_data_for_ship in all_ships_data.items():
        logger.info(f"Processing ship: {ship_id}")
        if "top_view" in views_data_for_ship and "side_view" in views_data_for_ship:
            logger.info(f"Found top and side views for {ship_id}. Proceeding to 3D framework.")
            apply_3d_framework_steps_for_ship(ship_id, views_data_for_ship)
        else:
            logger.warning(f"Skipping 3D framework for {ship_id}: Missing required views. "
                           f"Top view found: {'top_view' in views_data_for_ship}, "
                           f"Side view found: {'side_view' in views_data_for_ship}")

    logger.info("Main processing pipeline finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process ship images to generate 3D models.")
    parser.add_argument('--image_path', type=str, help='Path to a single image file.')
    parser.add_argument('--sample_dir', type=str, default=config.SAMPLE_IMAGE_DIR, help='Directory containing sample images.')
    parser.add_argument('--num_images', type=int, default=0, help='Number of images to process from sample_dir (0 for all).')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (e.g., for Gemini API calls).')
    
    args = parser.parse_args()
    main(args)
