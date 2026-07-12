# =============================================================================
# This module will be responsible for:
# 
# Building the current manifest
# Loading the previous manifest
# Saving the new manifest
# Identifying added files
# Identifying modified files
# Identifying deleted files
# =============================================================================
"""
Utilities for tracking changes to source documents.
"""

import logging
# Create a module-specific logger.
# __name__ helps identify which module generated a log message.
logger = logging.getLogger(__name__)

import hashlib
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS
from src.paths import (
    DATA_DIR,
    DOCUMENT_MANIFEST_PATH,
)

# Type alias describing the structure of a document manifest:
# {
#     "file_name.pdf": {
#         "size": int,
#         "modified_time_ns": int,
#         "sha256": str,
#     }
# }
Manifest = dict[str, dict[str, Any]]


def calculate_file_hash(
    file_path: Path,
    block_size: int = 65_536,
    ) -> str:
    """
    Calculate the SHA-256 hash of a file.

    The file is read in blocks so large documents do not need
    to be loaded entirely into memory.

    Args:
        file_path:
            Path to the file being hashed.

        block_size:
            Number of bytes read during each iteration.

    Returns:
        The hexadecimal SHA-256 hash of the file.
    """
    # Create an empty SHA-256 hash object that will be updated
    # incrementally as chunks of the file are read.
    sha256_hash = hashlib.sha256()

    # Open file in binary mode to support all file types
    # (PDFs, DOCX, TXT, etc.).
    with file_path.open("rb") as file:
    
        # Read the file in chunks rather than loading the entire
        # file into memory. The walrus operator (:=) assigns and
        # evaluates chunk in one step.
        while chunk := file.read(block_size):
            sha256_hash.update(chunk)
    
    # Return the final hexadecimal hash string.
    return sha256_hash.hexdigest()


def build_document_manifest(
    data_directory: Path = DATA_DIR,
    ) -> Manifest:
    """
    Build a manifest describing the current source documents.

    Each document is represented by its relative path, file size,
    modification time and SHA-256 content hash.

    Args:
        data_directory:
            Directory containing the source documents.

    Returns:
        A dictionary containing document metadata.
    """
    # Initialise an empty manifest that will hold metadata
    # for all supported documents.
    manifest: Manifest = {}
    
    # Recursively scan the data directory and all subdirectories.
    for file_path in sorted(data_directory.rglob("*")):
    
        # Skip directories and process files only.
        if not file_path.is_file():
            continue
    
        # Ignore unsupported file types so they do not trigger
        # unnecessary FAISS rebuilds.
        if (
            file_path.suffix.lower()
            not in SUPPORTED_DOCUMENT_EXTENSIONS
        ):
            logger.info(
                "Detected unsupported file(s): %s",
                file_path.name,
            )
            continue
    
        # Store paths relative to the data directory so the
        # manifest remains portable across machines and OSs.
        relative_path = file_path.relative_to(
            data_directory
        ).as_posix()
    
        # Retrieve file metadata from the operating system.
        file_statistics = file_path.stat()
    
        # Build the manifest entry for this document.
        manifest[relative_path] = {
    
            # File size in bytes.
            "size": file_statistics.st_size,
    
            # Last modification timestamp in nanoseconds.
            "modified_time_ns": file_statistics.st_mtime_ns,
    
            # Content fingerprint used to detect modifications.
            "sha256": calculate_file_hash(file_path),
        }
    
    # Return the completed manifest.
    return manifest


def load_saved_manifest(
    manifest_path: Path = DOCUMENT_MANIFEST_PATH,
    ) -> Manifest:
    """
    Load the previously saved document manifest.

    An empty dictionary is returned when the file does not exist,
    is empty, contains invalid JSON or has an unexpected structure.

    Args:
        manifest_path:
            Path to the saved manifest.

    Returns:
        The saved manifest or an empty dictionary.
    """

    # First run: no manifest has been saved yet.
    if not manifest_path.exists():
        return {}
    
    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
    
            # Convert JSON into a Python dictionary.
            manifest = json.load(file)
    
    # Return an empty manifest if the file is corrupt,
    # invalid JSON, or cannot be read.
    except (JSONDecodeError, OSError):
        return {}
    
    # Defensive check to ensure the loaded JSON has the
    # expected dictionary structure.
    if not isinstance(manifest, dict):
        return {}
    
    return manifest


def save_document_manifest(
    manifest: Manifest,
    manifest_path: Path = DOCUMENT_MANIFEST_PATH,
    ) -> None:
    """
    Save the document manifest as formatted JSON.

    Args:
        manifest:
            Current document manifest.

        manifest_path:
            Destination of the JSON manifest.
    """

    # Ensure the parent directory exists before saving.
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    # Write to a temporary file first to avoid leaving a
    # partially written manifest if the process crashes.
    temporary_path = manifest_path.with_suffix(
        ".json.tmp"
    )
    
    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
    
        # Save the manifest in a human-readable format.
        json.dump(
            manifest,
            file,
            indent=4,
            sort_keys=True,
        )
    
    # Atomically replace the old manifest with the new one.
    temporary_path.replace(manifest_path)


def identify_document_changes(
    current_manifest: Manifest,
    saved_manifest: Manifest,
) -> tuple[set[str], set[str], set[str]]:
    """
    Compare the current documents with the saved manifest.

    Args:
        current_manifest:
            Manifest generated from the current data directory.

        saved_manifest:
            Manifest generated during the previous index build.

    Returns:
        Three sets containing:

        1. Added files
        2. Modified files
        3. Deleted files
    """

    # Extract the document names from each manifest.
    current_files = set(current_manifest)
    saved_files = set(saved_manifest)
    
    # Files that exist now but did not exist before.
    added_files = current_files - saved_files
    
    # Files that existed before but no longer exist.
    deleted_files = saved_files - current_files
    
    # Files that exist in both manifests.
    common_files = current_files & saved_files
    
    # Common files whose content hash has changed.
    modified_files = {
        file_name
        for file_name in common_files
        if (
            current_manifest[file_name]["sha256"]
            != saved_manifest[file_name].get("sha256")
        )
    }
    
    return (
        added_files,
        modified_files,
        deleted_files,
    )