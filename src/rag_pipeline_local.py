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

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_vector_store(
    documents: list[Document]
    ) -> FAISS:
    
    """
    Create a FAISS vector store from loaded LangChain documents.

    Args:
        documents:
            List of LangChain Document objects loaded from source files.

    Returns:
        FAISS vector store containing embedded document chunks.
    """

    # Split documents into smaller overlapping chunks so retrieval
    # can return focused sections rather than entire documents.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(
        f"Created {len(chunks)} chunks"
    )

    # Use CUDA if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert text chunks into numerical vectors using a local
    # Sentence Transformer model from Hugging Face.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )

    # Store document chunks and their embeddings in FAISS so
    # similarity search can be performed efficiently.
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    logger.info(
        "FAISS vector store created"
    )

    return vector_store


def extract_sources(retrieved_docs: list[Document]) -> list[str]:
    """
    Extract unique source references from retrieved documents.

    Args:
        retrieved_docs: Documents returned by the retriever.

    Returns:
        A list of unique source references.
    """

    sources = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page_label") or doc.metadata.get("page")

        if page is not None:
            source_reference = f"{source} — page {page}"
        else:
            source_reference = source

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

    TOP_K = 4
    SIMILARITY_SCORE_THRESHOLD = 1.2

    # Retrieve the top matching document chunks with their FAISS distance scores.
    # Lower scores indicate stronger semantic similarity.
    docs_and_scores = vector_store.similarity_search_with_score(
        question,
        k=TOP_K
    )

    # Keep only chunks that are sufficiently relevant.
    relevant_docs = [doc for doc, score in docs_and_scores
                     if score < SIMILARITY_SCORE_THRESHOLD
                     ]

    if not relevant_docs:
        return "I do not know based on the available documents.", []

    sources = extract_sources(relevant_docs)

    document_information = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    llm = ChatOllama(
        model="llama3.1"
    )

    prompt = f"""
                You are a helpful document assistant.
                
                Answer the question using only the document information below.
                
                If the answer is not in the document, say you do not know.
                
                Document information:
                {document_information}
                
                Question:
                {question}
                
                Answer:
            """

    response = llm.invoke(prompt)

    return response.content, sources