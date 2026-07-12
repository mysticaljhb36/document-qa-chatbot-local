# =============================================================================
# This script handles file locations
# =============================================================================
"""
Centralised project paths.

This module defines reusable paths used throughout the
application and avoids hardcoded file locations.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
FAISS_METADATA_PATH = VECTOR_STORE_DIR / "index.pkl"
DOCUMENT_MANIFEST_PATH = (
    VECTOR_STORE_DIR / "document_manifest.json"
)





