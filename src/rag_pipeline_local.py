# =============================================================================
# This script handles embeddings + FAISS + Ollama answer generation
# =============================================================================

"""
RAG pipeline utilities.

This module handles the core local RAG pipeline:
    1. Split loaded documents into smaller chunks.
    2. Convert chunks into vector embeddings.
    3. Store embeddings in a FAISS vector index.
    4. Retrieve relevant chunks for a user question.
    5. Generate an answer using a local Ollama LLM.

Retrieval is handled with Sentence Transformers and FAISS.
Answer generation is handled locally with Ollama.
"""

import logging
import warnings
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama



from src.document_loader import load_documents_from_folder

from src.document_manifest import (
    build_document_manifest,
    identify_document_changes,
    load_saved_manifest,
    save_document_manifest,
)

from src.paths import (
    DATA_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    VECTOR_STORE_DIR,
)


from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    SIMILARITY_SCORE_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    LLM_MODEL_NAME
)

# Create a module-specific logger so log messages can be traced
# back to this file during debugging and monitoring.
logger = logging.getLogger(__name__)

# Suppress third-party deprecation warnings to keep logs clean.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialise the embedding model used for documents
    and user questions.

    Returns:
        Configured Hugging Face embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE
        },
    )


def create_vector_store(
    documents: list[Document],
    ) -> FAISS:
    """
    Create and persist a FAISS vector store.

    Args:
        documents:
            Loaded source documents.

    Returns:
        Newly created FAISS vector store.
    """

    # Split large documents into overlapping chunks so retrieval
    # can find relevant sections without exceeding model context limits.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    document_chunks = (
        text_splitter.split_documents(documents)
    )

    # Fail fast if no chunks were created rather than creating
    # an empty FAISS index.
    if not document_chunks:
        raise ValueError(
            "No document chunks were created. "
            "Check that the data directory contains "
            "supported documents."
        )

    embeddings = create_embedding_model()
    
    # Generate vector embeddings and build the FAISS index.
    vector_store = FAISS.from_documents(
        documents=document_chunks,
        embedding=embeddings,
    )

    # Persist the index so embeddings do not need to be rebuilt
    # every time the application starts.
    vector_store.save_local(
        str(VECTOR_STORE_DIR)
    )

    return vector_store


def vector_store_exists() -> bool:
    """
    Check whether both required FAISS files exist.

    Returns:
        True when index.faiss and index.pkl both exist.
    """

    return (
        FAISS_INDEX_PATH.is_file()
        and FAISS_METADATA_PATH.is_file()
    )


def load_vector_store() -> FAISS:
    """
    Load an existing FAISS vector store from disk.

    Returns:
        Loaded FAISS vector store.
    """

    embeddings = create_embedding_model()

    return FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vector_store() -> FAISS:
    """
    Load the existing FAISS index when documents have not
    changed, or rebuild it when changes are detected.

    Changes include:

    - A new document being added
    - An existing document being modified
    - An existing document being deleted
    - Missing FAISS index files
    - Missing or invalid document manifest

    Returns:
        A loaded or newly created FAISS vector store.
    """

    # Build the current view of documents and compare it with
    # the previously saved manifest.
    current_manifest = build_document_manifest()
    saved_manifest = load_saved_manifest()

    (
        added_files,
        modified_files,
        deleted_files,
    ) = identify_document_changes(
        current_manifest=current_manifest,
        saved_manifest=saved_manifest,
    )

    index_exists = vector_store_exists()

    # Convert change sets into a single boolean flag used
    # to decide whether a rebuild is required.
    documents_changed = bool(
        added_files
        or modified_files
        or deleted_files
    )
    
    # Stop execution if no supported source documents exist.
    if not current_manifest:
        raise FileNotFoundError(
            "No supported documents were found "
            "inside the data directory."
        )
        
    # Fast path: load the existing index when nothing has changed.    
    if index_exists and not documents_changed:
        logger.info(
            "No document changes detected. "
            "Loading the existing FAISS vector store."
        )

        return load_vector_store()
    
    # Missing FAISS files indicate the index must be rebuilt
    if not index_exists:
        logger.info(
            "A complete FAISS vector store was not found. "
            "Creating a new index."
        )

    if added_files:
        logger.info("New documents detected:")
    
        for file_name in sorted(added_files):
            logger.info(
                "New document detected: %s",
                file_name,
            )
    
    if modified_files:
        logger.info("Modified documents detected:")
    
        for file_name in sorted(modified_files):
            logger.info(
                "Modified document detected: %s",
                file_name,
            )
    
    if deleted_files:
        logger.info("Deleted documents detected:")
    
        for file_name in sorted(deleted_files):
            logger.info(
                "Deleted document detected: %s",
                file_name,
            )
    
    logger.info(
        "Rebuilding and saving the FAISS vector store."
    )
    
    logger.info("Current manifest: %s", current_manifest)
    logger.info("Saved manifest: %s", saved_manifest)
    logger.info("Added files: %s", added_files)
    logger.info("Modified files: %s", modified_files)
    logger.info("Deleted files: %s", deleted_files)
    
    logger.info(
        "Loading source documents from: %s",
        DATA_DIR,
    )
    
    # Load raw source documents prior to chunking and embedding.
    documents = load_documents_from_folder(DATA_DIR)
    
    logger.info(
        "Successfully loaded %d document objects.",
        len(documents),
    )
    
    logger.info("Creating the FAISS vector store.")
    
    # Rebuild and persist the vector store
    vector_store = create_vector_store(
        documents=documents
    )
    
    logger.info(
        "FAISS vector store created successfully."
    )
    
    # Save the current manifest so future runs can detect changes
    save_document_manifest(
        current_manifest
    )
    
    logger.info(
        "Document manifest saved successfully."
    )
    
    return vector_store


def extract_sources(
    retrieved_docs: list[Document],
) -> list[str]:
    """
    Extract unique source references from retrieved documents.

    Args:
        retrieved_docs:
            Document chunks returned by FAISS retrieval.

    Returns:
        Unique source references associated with the
        retrieved document chunks.
    """

    sources: list[str] = []

    for doc in retrieved_docs:

        metadata = doc.metadata or {}

        source = metadata.get(
            "source",
            "Unknown source",
        )

        # Convert pathlib.Path or other values to string.
        source = str(source)

        source_name = Path(source).name

        # Some document loaders provide page_label while others
        # provide page. Support both metadata formats.
        page = metadata.get("page_label")

        if page is None:
            page = metadata.get("page")

        if page is not None:
            source_reference = (
                f"{source_name} — page {page}"
            )
        else:
            source_reference = source_name
            
        # Prevent duplicate source references appearing in the UI.
        if source_reference not in sources:
            sources.append(source_reference)

    return sources

def generate_answer(
    question: str,
    vector_store: FAISS
) -> tuple[str, list[str]]:
    """
    Generate an answer to a user question using score-based RAG retrieval.

    Args:
        question: User question.
        vector_store: FAISS vector store containing embedded document chunks.

    Returns:
        A tuple containing:
            - Generated answer as a string.
            - List of source references used to support the answer.
    """

    # Retrieve the top matching document chunks with their FAISS distance scores.
    # Lower scores indicate stronger semantic similarity.
    docs_and_scores = vector_store.similarity_search_with_score(
        question,
        k=TOP_K
    )

    # Keep only chunks that are sufficiently relevant.
    relevant_docs = [
    doc
    for doc, score in docs_and_scores
    if score < SIMILARITY_SCORE_THRESHOLD
    ]

    if not relevant_docs:
        return "I do not know based on the available documents.", []
    
    # Extract source references for display alongside the answer.
    sources = extract_sources(relevant_docs)

    # Combine retrieved chunks into a single context block
    # that will be supplied to the LLM.
    document_information = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    llm = ChatOllama(
        model = LLM_MODEL_NAME
    )

    # Ground the model by restricting it to retrieved document content.
    prompt = f"""
    You are a helpful document assistant.
    
    Answer the question using only the document information below.
    
    If the answer is not in the document, say you do not know.
    
    Document information:
    {document_information}
    
    Question:
    {question}
    
    Answer:
            """.strip()

    response = llm.invoke(prompt)

    return response.content, sources