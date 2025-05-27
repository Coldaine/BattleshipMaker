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
    sys.path.append(PROJECT_ROOT)

from src.utils.logger import setup_logger # Explicitly import and call
from src.image_processing.loader import load_image_pil
from src.image_processing.splitter import split_image_if_needed
# Updated import from file_utils - Removed create_ship_image_output_directory
from src.utils.file_utils import get_filename_without_extension, sanitize_filename
from src import config
from src.gemini_api.client import analyze_image_with_gemini
from src.image_processing.vectorizer import vectorize_image_to_svg_subprocess, generate_hull_silhouette_mask
import base64
import numpy as np

logger = setup_logger()

# Helper function to get save path in flat type-based directories
# Moved outside the loop
def get_save_path_for_type(type_subdir_name, filename):
    base_output_dir = config.OUTPUT_DIR
    type_dir = os.path.join(base_output_dir, type_subdir_name)
    os.makedirs(type_dir, exist_ok=True)
    return os.path.join(type_dir, filename)

def apply_3d_framework_steps_for_ship(ship_id, views_data):
    logger.info(f"Starting 3D framework steps for {ship_id}")

    # --- Step 3: Scaling & Alignment ---
    logger.info("Step 3: Scaling & Alignment")
    # This is a simplified implementation based on ProposedChange.md
    # It assumes 'overall_length_L' is available in dimensions and refers to the hull.
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
                            break # Found the length, no need to check other dimensions/views
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

    # --- Step 5: Component Primitive Generation (Conceptual) ---
    logger.info("Step 5: Component Primitive Generation")
    # Now that we have scale factors and potentially hull contours in meters,
    # we can start generating 3D primitives for components.
    # This involves iterating through the components identified by Gemini in each view,
    # using their bounding boxes and the scale factors to determine their size and position in 3D space.
    # We'll need to reconcile the information from top and side views for each component.
    # This step will require a 3D geometry library.

    generated_components_3d_params = {}
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

                if top_comp and side_comp:
                    top_bbox_px = top_comp.get("bounding_box_2d")
                    side_bbox_px = side_comp.get("bounding_box_2d")

                    if top_bbox_px and side_bbox_px and scale_factors.get("top_view") is not None and scale_factors.get("side_view") is not None:
                        logger.debug(f"Calculating 3D parameters for {comp_name} from bounding boxes.")
                        try:
                            # Calculate 3D dimensions and center in meters using bboxes and scale_factors
                            # Mapping from 2D image coordinates to 3D world coordinates (assuming a right-handed Y-up system)
                            # Top view (X, Z): image X -> world X, image Y -> world Z
                            # Side view (X, Y): image X -> world X, image Y -> world Y

                            # Dimensions
                            width_m = (top_bbox_px[2] - top_bbox_px[0]) / scale_factors["top_view"]
                            height_m = (side_bbox_px[3] - side_bbox_px[1]) / scale_factors["side_view"]
                            depth_m = (top_bbox_px[3] - top_bbox_px[1]) / scale_factors["top_view"]

                            # Center position
                            # Reconcile X center from both views (e.g., average)
                            center_x_m_top = (top_bbox_px[0] + top_bbox_px[2]) / 2 / scale_factors["top_view"]
                            center_x_m_side = (side_bbox_px[0] + side_bbox_px[2]) / 2 / scale_factors["side_view"]
                            center_x_m = (center_x_m_top + center_x_m_side) / 2.0

                            center_y_m = (side_bbox_px[1] + side_bbox_px[3]) / 2 / scale_factors["side_view"]
                            center_z_m = (top_bbox_px[1] + top_bbox_px[3]) / 2 / scale_factors["top_view"]

                            primitive_dims_m = [width_m, height_m, depth_m]
                            primitive_center_m = [center_x_m, center_y_m, center_z_m]

                            generated_components_3d_params[comp_name] = {
                                "dimensions_m": primitive_dims_m,
                                "center_m": primitive_center_m,
                                "source_views": {"top_view": top_comp, "side_view": side_comp}
                            }
                            logger.info(f"Calculated 3D parameters for {comp_name}: Dims={primitive_dims_m}, Center={primitive_center_m}")

                            # --- Create 3D primitive object using a library (e.g., Open3D, Trimesh) ---
                            # This part requires a 3D geometry library and specific implementation.
                            # Example (conceptual): primitive_3d_object = create_box(center_m, dims_m)
                            # generated_components_3d_params[comp_name]["3d_object"] = primitive_3d_object # Store the 3D object

                        except Exception as e:
                            logger.error(f"Error calculating 3D parameters for {comp_name}: {e}", exc_info=True)
                            generated_components_3d_params[comp_name] = {"error": "Calculation failed"}
                    else:
                        logger.warning(f"Missing bounding box or scale factor for {comp_name} in one or both views. Cannot calculate 3D parameters.")
                        generated_components_3d_params[comp_name] = {"error": "Missing data"}
                else:
                    logger.warning(f"Missing data for {comp_name} in one or both views. Skipping.")
                    generated_components_3d_params[comp_name] = {"error": "Missing view data"}

        else: # Handle single object components like superstructure, bridge_or_tower
            top_comp = top_view_data.get(comp_type) if top_view_data else None
            side_comp = side_view_data.get(comp_type) if side_view_data else None

            if top_comp and side_comp:
                 top_bbox_px = top_comp.get("bounding_box_2d")
                 side_bbox_px = side_comp.get("bounding_box_2d")

                 if top_bbox_px and side_bbox_px and scale_factors.get("top_view") is not None and scale_factors.get("side_view") is not None:
                    logger.debug(f"Calculating 3D parameters for {comp_type} from bounding boxes.")
                    try:
                        # Calculate 3D dimensions and center in meters
                        width_m = (top_bbox_px[2] - top_bbox_px[0]) / scale_factors["top_view"]
                        height_m = (side_bbox_px[3] - side_bbox_px[1]) / scale_factors["side_view"]
                        depth_m = (top_bbox_px[3] - top_bbox_px[1]) / scale_factors["top_view"]

                        center_x_m_top = (top_bbox_px[0] + top_bbox_px[2]) / 2 / scale_factors["top_view"]
                        center_x_m_side = (side_bbox_px[0] + side_bbox_px[2]) / 2 / scale_factors["side_view"]
                        center_x_m = (center_x_m_top + center_x_m_side) / 2.0

                        center_y_m = (side_bbox_px[1] + side_bbox_px[3]) / 2 / scale_factors["side_view"]
                        center_z_m = (top_bbox_px[1] + top_bbox_px[3]) / 2 / scale_factors["top_view"]

                        primitive_dims_m = [width_m, height_m, depth_m]
                        primitive_center_m = [center_x_m, center_y_m, center_z_m]

                        generated_components_3d_params[comp_type] = {
                            "dimensions_m": primitive_dims_m,
                            "center_m": primitive_center_m,
                            "source_views": {"top_view": top_comp, "side_view": side_comp}
                        }
                        logger.info(f"Calculated 3D parameters for {comp_type}: Dims={primitive_dims_m}, Center={primitive_center_m}")

                        # --- Create 3D primitive object using a library (e.g., Open3D, Trimesh) ---\n                        # Example (conceptual): primitive_3d_object = create_box(center_m, dims_m)
                        # generated_components_3d_params[comp_type]["3d_object"] = primitive_3d_object # Store the 3D object

                    except Exception as e:
                        logger.error(f"Error calculating 3D parameters for {comp_type}: {e}", exc_info=True)
                        generated_components_3d_params[comp_type] = {"error": "Calculation failed"}
                 else:
                    logger.warning(f"Missing bounding box or scale factor for {comp_type} in one or both views. Cannot calculate 3D parameters.")
                    generated_components_3d_params[comp_type] = {"error": "Missing data"}
            else:
                logger.warning(f"Missing top or side view data for component: {comp_type}. Skipping.")
                generated_components_3d_params[comp_type] = {"error": "Missing view data"}

    # Store the calculated 3D parameters in the views_data (optional, could also return them)
    # For now, let's just log them and keep the structure flat.
    # In a real implementation, you'd likely have a dedicated ship_model data structure.
    views_data["generated_components_3d_params"] = generated_components_3d_params

    # --- Step 6: Hull Construction (Conceptual) ---
    logger.info("Step 6: Hull Construction (Conceptual)")
    # Use the vectorized hull contours in meters (from Step 4) to construct the 3D hull mesh.
    # This is a complex step involving lofting or skinning algorithms.
    # Requires a 3D library capable of these operations.

    # --- Step 7: 3D Grid / Point Cloud (Conceptual - Optional) ---
    logger.info("Step 7: 3D Grid / Point Cloud (Conceptual - Optional)")

    # --- Step 8: Assembly & Relative Positioning (Conceptual) ---
    logger.info("Step 8: Assembly & Relative Positioning (Conceptual)")
    # Combine the generated hull and component primitives into a single 3D model.
    # Position components relative to the hull based on their calculated 3D centers.

    # --- Step 9: Final AI Review (Conceptual) ---
    logger.info("Step 9: Final AI Review (Conceptual)")
    # Render the assembled 3D model from different angles.
    # Send the rendered images to Gemini for a final check on missing components, proportions, and placement.

    # --- Step 10: Export (Conceptual) ---
    logger.info("Step 10: Export (Conceptual)")
    # Save the final 3D model in a standard format (e.g., OBJ, STL).
    # Generate a report summarizing the process, identified components, dimensions, and any warnings.


    logger.info(f"Finished initial 3D framework steps for {ship_id}. Further steps require significant 3D geometry implementation.")


def process_single_image(image_path):
    logger.info(f"Starting processing for image: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"Input image not found: {image_path}")
        return

    img_pil = load_image_pil(image_path)
    if not img_pil:
        return

    split_images_pil = split_image_if_needed(img_pil)
    logger.info(f"Image split into {len(split_images_pil)} part(s).")

    source_filename_no_ext = get_filename_without_extension(image_path)
    # No longer need base_image_output_dir or first_ship_identification_for_dir at this scope
    # The base output directory is now config.OUTPUT_DIR, and type subdirs are handled in get_save_path_for_type

    # Dictionary to store analysis results per ship ID and view type
    ship_views_data = {}

    for i, view_pil in enumerate(split_images_pil):
        internal_view_identifier = f"{source_filename_no_ext}_view_{i+1}" # Used for logging & cache key
        logger.info(f"Processing view: {internal_view_identifier}")

        api_analysis_result = analyze_image_with_gemini(view_pil, internal_view_identifier)

        if not api_analysis_result:
            logger.error(f"Gemini API analysis failed for {internal_view_identifier}. Skipping further processing for this view.")
            continue

        # Extract top-level information
        current_ship_identification = api_analysis_result.get("ship_identification", "Unknown_Ship")
        view_type = api_analysis_result.get("view_type", "unknown_view")
        transcribed_text = api_analysis_result.get("transcribed_text", "")

        # Store the analysis result for this view, grouped by ship identification
        sanitized_current_ship_id = sanitize_filename(current_ship_identification)
        if sanitized_current_ship_id not in ship_views_data:
            ship_views_data[sanitized_current_ship_id] = {}
        
        # Store the Gemini analysis result
        ship_views_data[sanitized_current_ship_id][view_type] = api_analysis_result

        # Extract detailed component information (hull, superstructure, funnels, turrets)
        hull_info = api_analysis_result.get("hull", {})
        superstructure_info = api_analysis_result.get("superstructure", {})
        funnels_data = api_analysis_result.get("funnels", [])
        turrets_data = api_analysis_result.get("turrets", [])
        # bridge_info = api_analysis_result.get("bridge_or_tower", {}) # If needed later

        hull_mask_base64 = hull_info.get("segmentation_mask_base64_png") if hull_info else None
        superstructure_mask_base64 = superstructure_info.get("segmentation_mask_base64_png") if superstructure_info else None

        logger.info(f"Gemini Result for {internal_view_identifier}: Ship ID='{current_ship_identification}', View='{view_type}', Text Length={len(transcribed_text)}, "
                    f"Hull Mask Present={hull_mask_base64 is not None}, Superstructure Mask Present={superstructure_mask_base64 is not None}, "
                    f"Funnels detected={len(funnels_data)}, Turrets detected={len(turrets_data)}")

        view_suffix = f"view_{i+1}"
        new_file_base_name = f"{sanitized_current_ship_id}_{view_suffix}"

        # Save the original processed view image (split part)
        processed_view_filename = f"{new_file_base_name}_original_view.png"
        try:
            save_path = get_save_path_for_type("images", processed_view_filename)
            view_pil.save(save_path)
            logger.info(f"Saved original view to: {save_path}")
        except Exception as e:
            logger.error(f"Could not save original view {processed_view_filename}: {e}")

        # Save transcribed text
        if transcribed_text:
            text_filename = f"{new_file_base_name}_transcribed.txt"
            try:
                save_path = get_save_path_for_type("text", text_filename)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(transcribed_text)
                logger.info(f"Saved transcribed text to: {save_path}")
            except Exception as e:
                logger.error(f"Could not save transcribed text for {internal_view_identifier}: {e}")

        # --- Decode and Save Gemini Masks ---
        gemini_hull_mask_pil = None
        if hull_mask_base64:
            try:
                logger.debug(f"Decoding hull segmentation mask for {internal_view_identifier}")
                if "base64," in hull_mask_base64: hull_mask_base64 = hull_mask_base64.split("base64,",1)[1]
                mask_bytes = base64.b64decode(hull_mask_base64)
                gemini_hull_mask_pil = Image.open(BytesIO(mask_bytes))
                logger.debug(f"Value of gemini_hull_mask_pil IMMEDIATELY AFTER Image.open: {gemini_hull_mask_pil}")
                if gemini_hull_mask_pil:
                    debug_filename = f"{new_file_base_name}_debug_gemini_hull_mask.png"
                    save_path = get_save_path_for_type("images", debug_filename)
                    gemini_hull_mask_pil.save(save_path)
                    logger.info(f"Saved Gemini hull mask to: {save_path}")
            except Exception as e: logger.error(f"Error decoding/saving Gemini hull mask: {e}", exc_info=True); gemini_hull_mask_pil = None

        gemini_superstructure_mask_pil = None
        if superstructure_mask_base64:
            try:
                logger.debug(f"Decoding superstructure segmentation mask for {internal_view_identifier}")
                if "base64," in superstructure_mask_base64: superstructure_mask_base64 = superstructure_mask_base64.split("base64,",1)[1]
                mask_bytes = base64.b64decode(superstructure_mask_base64)
                gemini_superstructure_mask_pil = Image.open(BytesIO(mask_bytes))
                if gemini_superstructure_mask_pil:
                    debug_filename = f"{new_file_base_name}_debug_gemini_superstructure_mask.png"
                    save_path = get_save_path_for_type("images", debug_filename)
                    gemini_superstructure_mask_pil.save(save_path)
                    logger.info(f"Saved Gemini superstructure mask to: {save_path}")
            except Exception as e: logger.error(f"Error decoding/saving Gemini superstructure mask: {e}", exc_info=True); gemini_superstructure_mask_pil = None

        gemini_turret_masks_pil = []
        for idx, turret_item in enumerate(turrets_data):
            turret_mask_base64 = turret_item.get("segmentation_mask_base64_png")
            if turret_mask_base64:
                try:
                    logger.debug(f"Decoding turret {idx} segmentation mask for {internal_view_identifier}")
                    if "base64," in turret_mask_base64: turret_mask_base64 = turret_mask_base64.split("base64,",1)[1]
                    mask_bytes = base64.b64decode(turret_mask_base64)
                    turret_mask_pil = Image.open(BytesIO(mask_bytes))
                    if turret_mask_pil:
                        gemini_turret_masks_pil.append(turret_mask_pil)
                        debug_filename = f"{new_file_base_name}_debug_gemini_turret_{idx}_mask.png"
                        save_path = get_save_path_for_type("images", debug_filename)
                        turret_mask_pil.save(save_path)
                        logger.info(f"Saved Gemini turret {idx} mask to: {save_path}")
                except Exception as e: logger.error(f"Error decoding/saving Gemini turret {idx} mask: {e}", exc_info=True)

        logger.debug(f"Value of gemini_hull_mask_pil before check: {gemini_hull_mask_pil}")
        logger.debug(f"Truthiness of gemini_hull_mask_pil: {bool(gemini_hull_mask_pil)}")

        # Generate and save "Best Detail" SVG (from original view_pil)
        logger.info(f"Attempting to vectorize original view (best detail): {internal_view_identifier}")
        best_detail_svg_content = vectorize_image_to_svg_subprocess(view_pil)
        if best_detail_svg_content:
            svg_filename_detail = f"{new_file_base_name}_best_detail.svg"
            try:
                save_path = get_save_path_for_type("vector", svg_filename_detail)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(best_detail_svg_content)
                logger.info(f"Saved best detail SVG to: {save_path}")
            except Exception as e:
                logger.error(f"Could not save best detail SVG for {internal_view_identifier}: {e}")
        else:
            logger.warning(f"Best detail vectorization failed for view: {internal_view_identifier}")

        # Generate and save "Hull Shape Only" SVG
        # Priority: 1. Gemini Mask, 2. OpenCV-based silhouette generation
        hull_shape_source_pil = None
        source_description_for_log = ""
        mask_status_prefix = ""

        if gemini_hull_mask_pil:
            logger.info(f"Using Gemini-provided hull mask for hull shape SVG: {internal_view_identifier}")
            hull_shape_source_pil = gemini_hull_mask_pil
            source_description_for_log = "gemini_mask"
            mask_status_prefix = "gemini_mask"
            # Note: When using the Gemini mask, we skip the OpenCV generate_hull_silhouette_mask step.
            # The vectorize_image_to_svg_subprocess function is expected to handle the binary mask correctly.
        else:
            logger.info(f"Gemini hull mask not available. Generating hull silhouette mask using OpenCV for: {internal_view_identifier}")
            hull_shape_source_pil = generate_hull_silhouette_mask(view_pil, view_type=view_type)
            source_description_for_log = "opencv_silhouette"
            mask_status_prefix = "opencv_fallback"
            if hull_shape_source_pil:
                # Save the OpenCV generated mask for debugging/inspection (optional)
                mask_debug_filename = f"{new_file_base_name}_debug_opencv_hull_mask.png"
                try:
                    save_path = get_save_path_for_type("images", mask_debug_filename)
                    hull_shape_source_pil.save(save_path)
                    logger.info(f"Saved OpenCV debug hull silhouette mask to: {save_path}")
                except Exception as e:
                    logger.error(f"Could not save OpenCV debug hull mask for {internal_view_identifier}: {e}")
            else:
                logger.warning(f"OpenCV hull silhouette mask generation failed for view: {internal_view_identifier}")

        # Store the hull shape source image (mask) for later 3D processing
        if hull_shape_source_pil:
             if "processing_data" not in ship_views_data[sanitized_current_ship_id][view_type]:
                 ship_views_data[sanitized_current_ship_id][view_type]["processing_data"] = {}
             ship_views_data[sanitized_current_ship_id][view_type]["processing_data"]["hull_silhouette_mask_pil"] = hull_shape_source_pil

        if hull_shape_source_pil:
            logger.info(f"Attempting to vectorize hull shape ({source_description_for_log}) for: {internal_view_identifier}")
            hull_silhouette_svg_content = vectorize_image_to_svg_subprocess(hull_shape_source_pil)
            if hull_silhouette_svg_content:
                svg_filename_hull = f"{mask_status_prefix}_{new_file_base_name}_hull_shape_only.svg"
                try:
                    save_path = get_save_path_for_type("vector", svg_filename_hull)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(hull_silhouette_svg_content)
                    logger.info(f"Saved hull shape only SVG to: {save_path}")
                except Exception as e:
                    logger.error(f"Could not save hull shape only SVG for {internal_view_identifier}: {e}")
            else:
                logger.warning(f"Hull shape only vectorization failed for view ({source_description_for_log}): {internal_view_identifier}")

        # --- Generate "Structures Except Turrets" SVG using Gemini masks ---
        if gemini_superstructure_mask_pil and gemini_hull_mask_pil: # Need at least hull and superstructure
            logger.info(f"Attempting to generate 'structures_no_turrets' SVG using Gemini masks for {internal_view_identifier}")
            try:
                # Convert masks to NumPy arrays and ensure they are binary (0 or 255)
                # Gemini masks are typically 'P' mode (palette) or 'L' (grayscale). Convert to '1' (1-bit binary) for clarity.
                hull_np = np.array(gemini_hull_mask_pil.convert('1')) * 255
                superstructure_np = np.array(gemini_superstructure_mask_pil.convert('1')) * 255

                # Combine hull and superstructure
                combined_structure_np = np.logical_or(hull_np, superstructure_np).astype(np.uint8) * 255

                # Subtract turret masks
                if gemini_turret_masks_pil:
                    for turret_pil in gemini_turret_masks_pil:
                        if turret_pil.size == combined_structure_np.shape[::-1]: # Ensure turret mask matches main image size
                            turret_np = np.array(turret_pil.convert('1')) * 255
                            combined_structure_np = np.logical_and(combined_structure_np, np.logical_not(turret_np)).astype(np.uint8) * 255
                        else:
                            logger.warning(f"Turret mask size {turret_pil.size} does not match combined structure size {combined_structure_np.shape[::-1]} for {internal_view_identifier}. Skipping this turret.")

                structures_no_turrets_pil = Image.fromarray(combined_structure_np, mode='L')

                # Save debug mask
                debug_snt_filename = f"{new_file_base_name}_debug_gemini_structures_no_turrets_mask.png"
                save_path_snt_debug = get_save_path_for_type("images", debug_snt_filename)
                structures_no_turrets_pil.save(save_path_snt_debug)
                logger.info(f"Saved Gemini structures_no_turrets mask to: {save_path_snt_debug}")

                # Vectorize
                snt_svg_content = vectorize_image_to_svg_subprocess(structures_no_turrets_pil)
                if snt_svg_content:
                    snt_svg_filename = f"gemini_mask_{new_file_base_name}_structures_no_turrets.svg"
                    save_path_snt_svg = get_save_path_for_type("vector", snt_svg_filename)
                    with open(save_path_snt_svg, 'w', encoding='utf-8') as f:
                        f.write(snt_svg_content)
                    logger.info(f"Saved structures_no_turrets SVG to: {save_path_snt_svg}")
                else:
                    logger.warning(f"Vectorization of structures_no_turrets mask failed for {internal_view_identifier}")

            except Exception as e_snt:
                logger.error(f"Error generating 'structures_no_turrets' SVG for {internal_view_identifier}: {e_snt}", exc_info=True)
        elif gemini_superstructure_mask_pil or gemini_turret_masks_pil: # Log if some parts are missing for this new drawing
             logger.warning(f"Cannot generate 'structures_no_turrets' for {internal_view_identifier} due to missing hull or superstructure Gemini masks.")

    logger.info(f"Finished processing views for image: {image_path}")

    # --- Apply 3D Framework Steps per Ship ---
    for ship_id, views_data in ship_views_data.items():
        if "top_view" in views_data and "side_view" in views_data:
            logger.info(f"Applying 3D framework steps for ship: {ship_id}")
            apply_3d_framework_steps_for_ship(ship_id, views_data)
        else:
            missing_views = [view for view in ["top_view", "side_view"] if view not in views_data]
            logger.warning(f"Skipping 3D framework steps for ship {ship_id}: Missing data for views: {missing_views}")


def main():
    parser = argparse.ArgumentParser(description="Process 2D battleship images to generate 3D models.")
    parser.add_argument("--image_path", help="Path to a specific input image file. If provided, only this image is processed.", default=None)
    parser.add_argument("--sample_dir", default="Sample pictures/", help="Directory containing sample images to process if --image_path is not set.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to process from the sample_dir. Used only if --image_path is not set.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (overrides config).")
    args = parser.parse_args()

    if args.debug:
        config.DEBUG_FLAG = True
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled via command line.")

    if args.image_path:
        if not os.path.exists(args.image_path):
            logger.error(f"Specified image_path does not exist: {args.image_path}")
            return
        process_single_image(args.image_path)
    else:
        logger.info(f"No specific image_path provided. Processing up to {args.num_images} images from directory: {args.sample_dir}")
        if not os.path.isdir(args.sample_dir):
            logger.error(f"Sample directory not found: {args.sample_dir}")
            return

        supported_extensions = ["*.jpg", "*.jpeg", "*.png"]
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(args.sample_dir, ext)))

        if not image_files:
            logger.warning(f"No images found in {args.sample_dir} with supported extensions ({', '.join(supported_extensions)})")
            return

        image_files.sort()
        images_to_process = image_files[:args.num_images]
        logger.info(f"Found {len(image_files)} images. Will process the first {len(images_to_process)}: {images_to_process}")

        for image_file in images_to_process:
            process_single_image(image_file)

if __name__ == "__main__":
    main()
