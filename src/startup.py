# =============================================================================
# This script set up directories
# =============================================================================

from src.paths import (
    DATA_DIR,
    LOG_DIR,
    VECTOR_STORE_DIR
)

# Create directory if it does not already exist.
# exist_ok=True prevents errors when the directory
# already exists and does not overwrite its contents.
def create_directories() -> None:
    """
    Create required application directories.
    """

    directories = [
        DATA_DIR,
        LOG_DIR,
        VECTOR_STORE_DIR
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )