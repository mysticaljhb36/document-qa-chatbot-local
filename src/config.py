# =============================================================================
# This script handles configuration settings
# =============================================================================

"""
Application configuration settings.
"""
import torch

# Document types
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

# Embedding model
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-mpnet-base-v2"
)

# Embedding device
# Use CUDA if available, otherwise fall back to CPU.
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ollama model
LLM_MODEL_NAME = "llama3.1"

# Text splitting
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 4
SIMILARITY_SCORE_THRESHOLD = 1.2