import logging
import os
from datetime import datetime
from src import config

def setup_logger():
    """Sets up the main logger for the application."""
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(config.LOG_DIR, f"{config.LOG_FILE_PREFIX}{timestamp}.txt")

    logging.basicConfig(
        level=logging.DEBUG if config.DEBUG_FLAG else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler() # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logger initialized. Log file: {log_filename}")
    return logger

# Initialize logger when module is loaded
# logger = setup_logger() # This line can be uncommented if you want a global logger instance
                            # or keep it to be called explicitly from main.py
