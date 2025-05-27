from PIL import Image
from src import config
# from src.utils.logger import logger # Import if logger is globally initialized

def is_background_color(pixel_color_rgb):
    """Checks if a pixel color falls within the defined background color range."""
    r, g, b = pixel_color_rgb
    min_r, min_g, min_b = config.BACKGROUND_COLOR_MIN
    max_r, max_g, max_b = config.BACKGROUND_COLOR_MAX
    return (min_r <= r <= max_r and
            min_g <= g <= max_g and
            min_b <= b <= max_b)

def split_image_if_needed(image_pil):
    """Splits an image horizontally if a line between 35% and 65% of image height
    (checked at 5% intervals) is mostly background.

    Args:
        image_pil (Image.Image): The Pillow Image object.

    Returns:
        list[Image.Image]: A list containing one or two Pillow Image objects.
    """
    width, height = image_pil.size

    # Ensure image is in RGB mode to get pixel colors consistently
    rgb_image_pil = image_pil
    if image_pil.mode != 'RGB':
        rgb_image_pil = image_pil.convert('RGB')
        # logger.debug("Converted image to RGB for splitting check.")

    start_y_percent = 0.35
    end_y_percent = 0.65
    step_y_percent = 0.05

    # Calculate y coordinates to check
    y_coords_to_check = []
    current_percent = start_y_percent
    while current_percent <= end_y_percent:
        y_coords_to_check.append(int(height * current_percent))
        current_percent += step_y_percent
    
    # Ensure we don't check the same line twice if percentages align due to rounding
    y_coords_to_check = sorted(list(set(y_coords_to_check)))
    # And ensure lines are within image bounds (e.g., not at y=0 or y=height-1 if that causes issues for cropping)
    y_coords_to_check = [y for y in y_coords_to_check if 0 < y < height -1]

    # logger.debug(f"Splitting check: Will test y-coordinates: {y_coords_to_check} (range 35%-65% of height {height})")

    for middle_y in y_coords_to_check:
        background_pixel_count = 0
        for x in range(width):
            pixel_color = rgb_image_pil.getpixel((x, middle_y))
            if is_background_color(pixel_color):
                background_pixel_count += 1

        # logger.debug(f"Splitting check at y={middle_y}: {background_pixel_count}/{width} background pixels.")
        if background_pixel_count >= (width * config.SPLITTING_HEURISTIC_BACKGROUND_THRESHOLD):
            # logger.info(f"Splitting image horizontally at y={middle_y}.")
            # Use original image_pil for cropping to preserve original mode if not RGB
            top_image = image_pil.crop((0, 0, width, middle_y))
            bottom_image = image_pil.crop((0, middle_y, width, height))
            return [top_image, bottom_image]

    # logger.info("No suitable split line found within the 35%-65% range.")
    return [image_pil]
