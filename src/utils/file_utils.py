import os
import re
import shutil
import logging
from src import config

logger = logging.getLogger(__name__)

def sanitize_filename(name):
    """Sanitizes a string to be suitable for a filename."""
    name = re.sub(r'[^a-zA-Z0-9_\-. ]', '_', name) # Allow spaces, alphanumeric, underscore, hyphen, dot
    name = re.sub(r'\s+', '_', name) # Replace multiple spaces with single underscore
    return name

def get_filename_without_extension(filepath):
    """Extracts the filename without its extension from a full path."""
    return os.path.splitext(os.path.basename(filepath))[0]
