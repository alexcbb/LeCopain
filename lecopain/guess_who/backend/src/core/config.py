# src/core/config.py
import logging.config
import os

# Get the base directory of the project (assuming config.py is in src/core)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGGING_CONFIG_PATH = os.path.join(BASE_DIR, 'logging.conf')

# Configure logging
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
    logging.getLogger(__name__).info("Logging configured from file.")
else:
    # Basic config if file is missing (though main.py has a fallback too)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"Logging configuration file not found at {LOGGING_CONFIG_PATH}. Using basicConfig.")

# Other configurations can go here
# SECRET_KEY = "your-secret-key" # Example