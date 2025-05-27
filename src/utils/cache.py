import os
import diskcache
from src import config
import logging

logger = logging.getLogger(__name__)

# Ensure cache directory exists
if not os.path.exists(config.CACHE_DIR):
    try:
        os.makedirs(config.CACHE_DIR)
        logger.info(f"Created cache directory: {config.CACHE_DIR}")
    except OSError as e:
        logger.error(f"Error creating cache directory {config.CACHE_DIR}: {e}")
        # Depending on desired behavior, could raise an error or disable caching

# Initialize cache
# The cache directory must exist before Cache is instantiated.
cache = diskcache.Cache(config.CACHE_DIR, expire=config.CACHE_EXPIRATION_SECONDS)

def get_from_cache(key):
    """Retrieves an item from the cache."""
    try:
        result = cache.get(key)
        if result is not None:
            logger.debug(f"Cache HIT for key: {key}")
        else:
            logger.debug(f"Cache MISS for key: {key}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving from cache for key {key}: {e}")
        return None

def set_to_cache(key, value):
    """Sets an item in the cache."""
    try:
        cache.set(key, value)
        logger.debug(f"Stored in cache with key: {key}")
    except Exception as e:
        logger.error(f"Error setting to cache for key {key}: {e}")

def clear_cache():
    """Clears the entire cache."""
    try:
        count = cache.clear()
        logger.info(f"Cache cleared. {count} items removed.")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    # This part will only run if cache.py is executed directly
    # For this to work standalone without main.py logger, setup a temporary one:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("cache_test")

    logger.info(f"Cache directory: {config.CACHE_DIR}")
    logger.info(f"Cache expiration: {config.CACHE_EXPIRATION_SECONDS} seconds")

    test_key = "my_test_key"
    test_value = {"data": "This is a test"}

    logger.info(f"Setting value for key: {test_key}")
    set_to_cache(test_key, test_value)

    logger.info(f"Getting value for key: {test_key}")
    retrieved_value = get_from_cache(test_key)
    logger.info(f"Retrieved value: {retrieved_value}")

    assert retrieved_value == test_value

    # clear_cache() # Uncomment to test clearing
