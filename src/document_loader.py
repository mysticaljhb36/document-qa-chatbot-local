# =============================================================================
# This script handles text extraction from data folder path
# =============================================================================
"""
Document loading utilities.

This module loads supported document types and converts them
into LangChain Document objects for downstream processing
such as chunking, embeddings and vector storage.

Keeping document loading separate from the RAG pipeline makes
the code easier to test, maintain and reuse.
"""
import logging
# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


def load_documents_from_folder(folder_path: str | Path):
    """
    Load all supported documents from a folder.

    Supported formats:
        - PDF
        - DOCX
        - TXT

    Args:
        folder_path:
            Directory containing documents to load.

    Returns:
        A flat list of LangChain Document objects.
    """

    documents = []

    # Convert string paths into Path objects so pathlib methods
    # can be used consistently throughout the function.
    folder = Path(folder_path)

    # Validate early that folder exist and folder is a directory
    # As we don't want to read unnecessary folders 
    if not folder.exists():
        logger.error(
        f"Folder not found: {folder}"
        )
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        logger.error(
        f"Expected a folder, got: {folder}"
    )
        raise NotADirectoryError(f"Expected a folder, got: {folder}")

    # Discover files dynamically so new documents can be added
    # to the folder without changing the source code.
    for file in folder.iterdir():        
        # Skip subfolders because this loader currently handles files 
        # in folder_path only.
        if not file.is_file():        
            continue 

        # Select the correct LangChain loader for each supported file type.
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            print(file)

        elif file.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file))

        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")

        else:
            # Unsupported files i.e image.png shouldn't crash the full pipeline.
            continue

        try:
            # Loading can fail for reasons we cannot fully predict,
            # such as corrupted PDFs, encrypted files or encoding issues.
            loaded_documents = loader.load()

        except Exception as error:
            logger.warning(
                f"Failed to load {file.name}: {error}"
            )
        
            continue

        # Add source metadata so retrieved chunks can be traced
        # back to the original file later.
        for document in loaded_documents:
            document.metadata["source"] = file.name

        # loader.load() returns a list, so extend keeps one flat list.
        documents.extend(loaded_documents)

    return documents

