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





