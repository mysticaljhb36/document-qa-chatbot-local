# =============================================================================
# Packages Setup
# =============================================================================
"""
Central logging configuration for the machine learning pipeline.

This module:

1. Creates the logs directory if it does not exist.
2. Configures application-wide logging.
3. Writes log messages to pipeline.log.
4. Preserves existing logs by using append mode.
5. Provides a reusable logger object.
"""

import logging

from src.paths import LOG_DIR


# Create logs directory if it does not already exist.
# exist_ok=True prevents errors when the directory
# already exists and does not overwrite its contents.
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Configure application-wide logging.
logging.basicConfig(

    # Persist logs to disk for monitoring and troubleshooting.
    filename=LOG_DIR / "pipeline.log",

    # Append to existing log file instead of overwriting.
    filemode="a",

    # Record INFO and higher-severity events.
    level=logging.INFO,

    # Include timestamp, module name and severity level.
    format=(
        "%(asctime)s - "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    ),

    # Override any logging configuration created by
    # Jupyter, Spyder or imported libraries.
    force=True
)

# Suppress unncessary third party logs
for noisy_logger in [
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "faiss",
    "faiss.loader"
    ]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Create reusable module logger.
loggers = logging.getLogger(__name__)


if __name__ == "__main__":

    loggers.info(
        "Logging configuration initialised successfully."
    )
    

