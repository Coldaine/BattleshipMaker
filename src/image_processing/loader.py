from PIL import Image
# from src.utils.logger import logger # Import if logger is globally initialized

def load_image_pil(image_path):
    """Loads an image using Pillow.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image.Image: The Pillow Image object, or None if loading fails.
    """
    try:
        img = Image.open(image_path)
        # logger.info(f"Successfully loaded image: {image_path}")
        return img
    except FileNotFoundError:
        # logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        # logger.error(f"Error loading image {image_path}: {e}")
        return None
