# src/image_processing/vectorizer.py
import cv2
import numpy as np
from PIL import Image
# import potrace # No longer using pypotrace library
import os
import logging
import subprocess # For calling potrace.exe
import tempfile # For creating temporary files

logger = logging.getLogger(__name__)

def pil_to_cv2(pil_image):
    """Converts a PIL Image to an OpenCV image (NumPy array)."""
    # Convert PIL image to NumPy array
    # If PIL image is RGBA, convert to RGB first to avoid issues with some OpenCV functions
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode == 'L': # Grayscale
        pass # Already suitable for conversion
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB') # Default to RGB if other mode

    return np.array(pil_image)

def binarize_image(cv2_image, threshold_value=127, max_value=255, invert=False):
    """Converts an image to binary (black and white) using a threshold.

    Args:
        cv2_image (np.array): OpenCV image.
        threshold_value (int): Pixel value threshold.
        max_value (int): Value to assign to pixels above the threshold.
        invert (bool): If True, use THRESH_BINARY_INV (object becomes white, bg black).
                       If False, use THRESH_BINARY (object becomes black, bg white).

    Returns:
        np.array: Binarized OpenCV image.
    """
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3: # Check if it's a color image
        gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    elif len(cv2_image.shape) == 2: # Already grayscale
        gray_image = cv2_image
    else:
        logger.error("Unsupported image format for binarization. Expected grayscale or BGR.")
        raise ValueError("Unsupported image format for binarization.")

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary_image = cv2.threshold(gray_image, threshold_value, max_value, threshold_type)
    return binary_image

# --- New Helper Functions for Side View Hull Refinement ---

def find_first_dark_pixel_y_from_top(cv2_image_binary_inverted, scan_step=1):
    """Scans from top to find the first row containing a dark (object) pixel.
    Assumes cv2_image_binary_inverted has the object as white (255) and background as black (0).
    """
    height, width = cv2_image_binary_inverted.shape
    for y in range(0, height, scan_step):
        if np.any(cv2_image_binary_inverted[y, :]): # Check if any pixel in the row is non-zero (white)
            logger.debug(f"First dark pixel (object) found at y={y} from top.")
            return y
    logger.debug("No dark pixels (object) found by find_first_dark_pixel_y_from_top.")
    return None # Should not happen if there's an object

def analyze_contour_complexity(contour_points, epsilon_factor=0.005, max_points_for_simple=15):
    """Analyzes a contour for simplicity (e.g., 'one bend').
    This is a placeholder and needs significant refinement.
    Args:
        contour_points: NumPy array of contour points.
        epsilon_factor: Factor for cv2.approxPolyDP related to arc length.
        max_points_for_simple: Max number of points after approximation to be considered simple.
    Returns:
        bool: True if the contour is considered simple, False otherwise.
    """
    if contour_points is None or len(contour_points) < 2:
        return False # Not a valid line
    
    # Approximate the contour to simplify it
    perimeter = cv2.arcLength(contour_points, True)
    approx_poly = cv2.approxPolyDP(contour_points, epsilon_factor * perimeter, True)
    num_points = len(approx_poly)
    logger.debug(f"Contour complexity: original points={len(contour_points)}, approximated points={num_points}")

    # Basic heuristic: fewer points after approximation might mean simpler curve.
    # This is a very rough heuristic for "one bend" and will need improvement.
    if num_points <= max_points_for_simple:
        logger.debug(f"Contour considered simple (<= {max_points_for_simple} points after approximation).")
        return True
    else:
        logger.debug(f"Contour considered complex (> {max_points_for_simple} points after approximation).")
        return False

def attempt_deck_line_cut(pil_image, initial_y_offset=5, y_step=2, max_iterations=50):
    """Attempts to find a clean deck line by iteratively cutting from the top.

    Args:
        pil_image (PIL.Image.Image): The input side-view image.
        initial_y_offset (int): How many pixels below the first detected ship pixel to start cutting.
        y_step (int): How many pixels to cut more in each iteration.
        max_iterations (int): Max number of cuts to try.

    Returns:
        int: The y-coordinate for the deck line cut if a simple line is found, otherwise None.
    """
    logger.info("Attempting to find deck line cut for side view.")
    cv2_img_original_color = pil_to_cv2(pil_image)
    if cv2_img_original_color is None or len(cv2_img_original_color.shape) < 2:
        logger.error("Failed to convert PIL image to CV2 format or invalid image for deck line cut.")
        return None
    height, width = cv2_img_original_color.shape[:2]

    # Binarize for finding the top of the ship (object white 255, bg black 0)
    binary_for_scan = binarize_image(cv2_img_original_color, invert=True)
    
    first_ship_pixel_y = find_first_dark_pixel_y_from_top(binary_for_scan)
    if first_ship_pixel_y is None:
        logger.warning("Could not find top of ship in side view for deck line cut.")
        return None

    start_y_cut = first_ship_pixel_y + initial_y_offset

    for i in range(max_iterations):
        current_y_cut = start_y_cut + (i * y_step)
        if current_y_cut >= height - 10: # Don't cut too much (leave at least 10px height)
            logger.debug("Deck line cut reached near bottom of image, stopping.")
            break

        logger.debug(f"Testing deck line cut at y={current_y_cut}")
        # Create a temporary image cropped from current_y_cut downwards
        # We need to find contours on this cropped part.
        # Binarize the original image (object white, bg black for findContours)
        binary_for_contours = binarize_image(cv2_img_original_color, invert=True)
        
        # Take the part of the binarized image from the cut downwards
        cropped_binary_for_contours = binary_for_contours[current_y_cut:, :]
        if cropped_binary_for_contours.shape[0] < 5: # If remaining image is too small
            logger.debug("Cropped image too small to analyze for deck line.")
            continue

        contours, _ = cv2.findContours(cropped_binary_for_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug("No contours found in cropped section.")
            continue

        # Find the contour that is most likely the "top line" of the remaining ship part.
        # This is heuristic: could be the one with the smallest average y in the *cropped* image's coordinates.
        # Or, the one whose bounding box starts highest.
        # For simplicity, let's find the highest point of any contour.
        # A more robust method would be to trace the top edge of the largest connected component.
        
        # Let's try to get the top-most contour based on its bounding box in the cropped image
        top_most_contour = None
        min_contour_y_in_cropped = float('inf')

        for c in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(c)
            if y_c < min_contour_y_in_cropped:
                min_contour_y_in_cropped = y_c
                top_most_contour = c
            elif y_c == min_contour_y_in_cropped: # If multiple contours start at same top line
                if top_most_contour is None or cv2.contourArea(c) > cv2.contourArea(top_most_contour):
                    top_most_contour = c # Prefer larger one if at same height
        
        if top_most_contour is None:
            logger.debug("Could not identify a top-most contour in the cropped section.")
            continue

        # The contour points are relative to the cropped image. We need to analyze its shape.
        # The actual "deck line" is the top edge of this top_most_contour.
        # For now, let's analyze the complexity of the whole top_most_contour as a proxy.
        if analyze_contour_complexity(top_most_contour):
            logger.info(f"Found potentially simple deck line at y_cut={current_y_cut} (original image coordinates).")
            return current_y_cut

    logger.warning("Could not find a simple deck line after trying multiple cuts.")
    return None

# --- End of New Helper Functions ---

def generate_hull_silhouette_mask(pil_image, view_type="unknown", 
                                  top_view_close_kernel_size=(25,25), top_view_open_kernel_size=(7,7),
                                  side_view_close_kernel_size=(20,5), side_view_open_kernel_size=(5,3)):
    """Generates a PIL image that is a solid silhouette of the largest object.
    For side views, attempts to find a clean deck line and crops the image before processing.
    """
    logger.info(f"Generating hull silhouette mask for view_type: {view_type}.")
    
    current_close_kernel_size = top_view_close_kernel_size
    current_open_kernel_size = top_view_open_kernel_size

    if view_type == "side_view":
        current_close_kernel_size = side_view_close_kernel_size
        current_open_kernel_size = side_view_open_kernel_size
        logger.info(f"Using side view kernels: CLOSE={current_close_kernel_size}, OPEN={current_open_kernel_size}")
    else: 
        if view_type != "top_view":
            logger.warning(f"Unknown view_type '{view_type}', defaulting to top_view morphological parameters.")
        logger.info(f"Using top view kernels: CLOSE={current_close_kernel_size}, OPEN={current_open_kernel_size}")

    try:
        image_to_process_pil = pil_image 

        if view_type == "side_view":
            logger.info("Attempting deck line cut for side view before morphological operations.")
            deck_line_y = attempt_deck_line_cut(pil_image)
            if deck_line_y is not None and deck_line_y < pil_image.height - 5: # Ensure cut is not too low
                logger.info(f"Applying deck line cut at y={deck_line_y} for side view silhouette generation.")
                image_to_process_pil = pil_image.crop((0, deck_line_y, pil_image.width, pil_image.height))
            else:
                logger.info("No suitable deck line cut found or cut too low. Proceeding with full side view for silhouette.")

        cv2_img = pil_to_cv2(image_to_process_pil) 
        if cv2_img is None or cv2_img.size == 0:
            logger.error("Image for silhouette generation is empty or invalid after potential crop.")
            return None

        binary_for_contours = binarize_image(cv2_img, invert=True) 

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, current_close_kernel_size)
        closed_img = cv2.morphologyEx(binary_for_contours, cv2.MORPH_CLOSE, close_kernel)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, current_open_kernel_size)
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, open_kernel)

        contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found after morphological operations.")
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(opened_img)
        cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
        
        silhouette_pil_image = Image.fromarray(mask, mode='L') 
        logger.info("Successfully generated hull silhouette mask.")
        return silhouette_pil_image

    except Exception as e:
        logger.error(f"Error generating hull silhouette mask: {e}", exc_info=True)
        return None

def vectorize_image_to_svg_subprocess(pil_image, turdsize=2, opttolerance=0.2, potrace_executable="potrace"):
    """Converts a PIL image to an SVG string using potrace.exe via subprocess.
       Handles general images by binarizing them, and pre-binarized masks by ensuring correct polarity for Potrace.
    """
    logger.info(f"Starting vectorization using subprocess. Turdsize: {turdsize}, Opttolerance: {opttolerance}")
    
    temp_input_file = None
    temp_output_file = None

    try:
        cv2_img_input = pil_to_cv2(pil_image)
        if cv2_img_input is None:
            logger.error("Failed to convert PIL image to CV2 format for vectorization.")
            return None

        # Prepare image for Potrace: Potrace expects black features (0) on a white background (255).
        binary_for_potrace_bmp = None

        # Check if the input is likely a binary mask (e.g., from Gemini or our silhouette function)
        # These masks are typically white object (255) on black background (0).
        if cv2_img_input.ndim == 2 or (cv2_img_input.ndim == 3 and cv2_img_input.shape[2] == 1):
            # If it's grayscale or single channel, assume it might be a mask.
            # Masks from Gemini (decoded PNG) or our silhouette function are typically white object on black background.
            # We need to invert it for Potrace: black object on white background.
            logger.debug("Input image is grayscale/single-channel, assuming it's a mask (white object on black bg). Inverting for Potrace.")
            binary_for_potrace_bmp = cv2.bitwise_not(cv2_img_input) 
        elif cv2_img_input.ndim == 3 and cv2_img_input.shape[2] == 3: # Color image
            logger.debug("Input image is color. Performing standard binarization (dark features to black).")
            binary_for_potrace_bmp = binarize_image(cv2_img_input, invert=False) # Object black, bg white
        else:
            logger.error(f"Unsupported image format for vectorization input: {cv2_img_input.shape}")
            return None

        # Ensure it's a 2D array for imwrite if it became 3D with one channel
        if binary_for_potrace_bmp.ndim == 3 and binary_for_potrace_bmp.shape[2] == 1:
            binary_for_potrace_bmp = binary_for_potrace_bmp.squeeze(axis=2)

        # 3. Save the binarized image to a temporary BMP file (Potrace handles BMP well)
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp_bmp:
            temp_input_file = tmp_bmp.name
        cv2.imwrite(temp_input_file, binary_for_potrace_bmp)
        logger.debug(f"Saved temporary binarized BMP to: {temp_input_file}")

        # 4. Prepare to save SVG output to another temporary file
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            temp_output_file = tmp_svg.name
        
        # 5. Construct and run the Potrace command
        # Command: potrace <input.bmp> -s -o <output.svg> -t <turdsize> -O <opttolerance>
        # -s means SVG output
        command = [
            potrace_executable,
            temp_input_file,
            "-s",  # Output format SVG
            "-o", temp_output_file,
            "-t", str(turdsize),
            "-O", str(opttolerance)
        ]
        logger.debug(f"Executing Potrace command: {' '.join(command)}")
        
        # On Windows, it's good practice to use shell=False and pass command as a list.
        # Ensure potrace_executable is either "potrace" (if in PATH) or the full path to potrace.exe.
        process = subprocess.run(command, capture_output=True, text=True, check=False) # check=False to handle errors manually

        if process.returncode != 0:
            logger.error(f"Potrace execution failed with return code {process.returncode}")
            logger.error(f"Potrace stderr: {process.stderr}")
            logger.error(f"Potrace stdout: {process.stdout}")
            return None
        
        logger.info("Potrace execution successful.")

        # 6. Read the generated SVG content
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        logger.info("Vectorization to SVG string successful via subprocess.")
        return svg_content

    except FileNotFoundError as e:
        logger.error(f"Potrace executable not found at '{potrace_executable}'. Ensure it's in PATH or provide full path. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during vectorization with subprocess: {e}", exc_info=True)
        return None
    finally:
        # 7. Clean up temporary files
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.remove(temp_input_file)
                logger.debug(f"Removed temporary input file: {temp_input_file}")
            except Exception as e_rem_in:
                logger.warning(f"Could not remove temporary input file {temp_input_file}: {e_rem_in}")
        if temp_output_file and os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
                logger.debug(f"Removed temporary output file: {temp_output_file}")
            except Exception as e_rem_out:
                logger.warning(f"Could not remove temporary output file {temp_output_file}: {e_rem_out}")

# Alias the new function to the old name for compatibility if main.py was already updated
vectorize_image_to_svg = vectorize_image_to_svg_subprocess

if __name__ == '__main__':
    # This is for direct testing of the vectorizer module
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("Vectorizer module direct test started.")

    # Create a dummy PIL image for testing (e.g., a black square on white background)
    # test_image_pil = Image.new('RGB', (100, 100), 'white')
    # from PIL import ImageDraw
    # draw = ImageDraw.Draw(test_image_pil)
    # draw.rectangle([20, 20, 80, 80], fill='black')
    
    # Or load an actual image
    # Ensure this path is correct if running directly
    # The CWD for direct execution of this script is src/image_processing/
    sample_image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Sample pictures"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))

    sample_image_filename = "north-carolina-class-battleship-recognition-drawings-b133c9-640.jpg"
    # sample_image_filename = "konig-class-battleship-janes-fighting-ships-1919-project-gutenberg-etext-24797-d1aa2c.png"

    # Try to use a processed image if available, as it's already split
    processed_image_path_test = os.path.join(output_dir, "Iowa-class_battleship", "north-carolina-class-battleship-recognition-drawings-b133c9-640", "north-carolina-class-battleship-recognition-drawings-b133c9-640_view_1_processed.png")

    if os.path.exists(processed_image_path_test):
        sample_image_path = processed_image_path_test
        logger.info(f"Using processed image for testing: {sample_image_path}")
    elif os.path.exists(os.path.join(sample_image_dir, sample_image_filename)):
        sample_image_path = os.path.join(sample_image_dir, sample_image_filename)
        logger.info(f"Using sample image for testing: {sample_image_path}")
    else:
        sample_image_path = None # Fallback to dummy

    if not sample_image_path or not os.path.exists(sample_image_path):
        logger.error(f"Test image not found at expected paths. Creating dummy image.")
        # Create a dummy image: white background, black square in the middle
        test_image_pil = Image.new('RGB', (100, 100), 'white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image_pil)
        draw.rectangle([20, 20, 80, 80], fill='black') # Ensure ship is black for THRESH_BINARY
        img_identifier = "dummy_square"
    else:
        logger.info(f"Loading test image: {sample_image_path}")
        test_image_pil = Image.open(sample_image_path)
        img_identifier = os.path.splitext(os.path.basename(sample_image_path))[0]

    if test_image_pil:
        logger.info("Attempting to vectorize test image using subprocess...")
        # If potrace is not in PATH, you might need to specify the full path:
        # e.g., svg_output = vectorize_image_to_svg_subprocess(test_image_pil, potrace_executable="C:\\Program Files\\Potrace\\potrace.exe")
        svg_output = vectorize_image_to_svg_subprocess(test_image_pil)
        if svg_output:
            output_filename = f"{img_identifier}_vectorized_subprocess.svg"
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            final_output_path = os.path.join(output_dir, output_filename)
            
            with open(final_output_path, 'w') as f:
                f.write(svg_output)
            logger.info(f"Successfully vectorized and saved to {final_output_path}")
        else:
            logger.error("Vectorization failed for the test image using subprocess.")
    else:
        logger.error("Failed to load the test image.")

    # Test the new silhouette generation
    if test_image_pil:
        logger.info("Attempting to generate hull silhouette mask...")
        # Simulate a side view for testing the new cut logic if needed
        # view_type_for_test = "side_view" 
        view_type_for_test = "top_view" # or "side_view" to test different kernels
        silhouette_mask_pil = generate_hull_silhouette_mask(test_image_pil, view_type=view_type_for_test)
        
        if silhouette_mask_pil:
            silhouette_mask_pil.save(os.path.join(output_dir, f"{img_identifier}_silhouette_mask_({view_type_for_test}).png"))
            logger.info(f"Saved silhouette mask to {output_dir}/{img_identifier}_silhouette_mask_({view_type_for_test}).png")

            refined_silhouette_pil = silhouette_mask_pil
            if view_type_for_test == "side_view": # Apply refinement only for side views
                logger.info("Attempting to apply deck line cut for side view test...")
                refined_silhouette_pil = attempt_deck_line_cut(silhouette_mask_pil)
                if refined_silhouette_pil != silhouette_mask_pil: # Check if it was actually modified
                    refined_silhouette_pil.save(os.path.join(output_dir, f"{img_identifier}_refined_side_silhouette_mask.png"))
                    logger.info(f"Saved refined side silhouette mask to {output_dir}/{img_identifier}_refined_side_silhouette_mask.png")
                else:
                    logger.info("Deck line cut did not modify the silhouette.")

            logger.info("Attempting to vectorize potentially refined silhouette mask...")
            svg_silhouette_output = vectorize_image_to_svg_subprocess(refined_silhouette_pil)
            if svg_silhouette_output:
                silhouette_svg_filename = f"{img_identifier}_vectorized_hull_silhouette_({view_type_for_test}).svg"
                final_silhouette_svg_path = os.path.join(output_dir, silhouette_svg_filename)
                with open(final_silhouette_svg_path, 'w') as f:
                    f.write(svg_silhouette_output)
                logger.info(f"Successfully vectorized silhouette and saved to {final_silhouette_svg_path}")
            else:
                logger.error("Vectorization of silhouette mask failed.")
        else:
            logger.error("Hull silhouette mask generation failed.")
        
        # Test original "best detail" vectorization (optional here, as it was tested before)
        # logger.info("Attempting to vectorize test image (best detail) using subprocess...")
        # svg_output = vectorize_image_to_svg_subprocess(test_image_pil)
        # ... (rest of the original test for best detail)
    
    logger.info("Vectorizer module direct test finished.")
