import os # For os.getenv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration for the Battleship 2D to 3D Conversion Project

# Gemini API Settings
GEMINI_API_MODEL = "gemini-1.5-flash-latest" # Ensuring this is the intended latest flash model
# API Key will be loaded from .env file (GOOGLE_API_KEY) by load_dotenv()
# and picked up by the google-generativeai library automatically.
# The config.GEMINI_API_KEY variable is not strictly needed by the Gemini client if GOOGLE_API_KEY is set.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "") # Fallback to empty string if not found

# File Paths
OUTPUT_DIR = "output/"
LOG_DIR = "logs/" # Suggesting a dedicated log directory
LOG_FILE_PREFIX = "run_log_"
SAMPLE_IMAGE_DIR = "e:\\A Unity projects\\Battleship maker\\Sample pictures/"

# Debugging
DEBUG_FLAG = True # Set to True to force API calls even if cache exists, and for more verbose logging

# Image Processing
SPLITTING_HEURISTIC_BACKGROUND_THRESHOLD = 0.95 # Percentage of pixels in middle line that must be background to split
# Define background color range (e.g., for white background)
# This might need to be more sophisticated, e.g. a function or a list of RGB tuples
BACKGROUND_COLOR_MIN = (240, 240, 240) 
BACKGROUND_COLOR_MAX = (255, 255, 255)
USE_OPENCV_FALLBACK_FOR_HULL_MASK = True
VECTORIZE_HULL_MASK_FALLBACK = True # Added missing configuration

# Caching
CACHE_DIR = ".cache/"
CACHE_EXPIRATION_SECONDS = 7 * 24 * 60 * 60 # 7 days
