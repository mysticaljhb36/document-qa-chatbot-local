# =============================================================================
# This script handles embeddings + FAISS
# =============================================================================

"""
RAG pipeline utilities.

This module handles the core retrieval pipeline:
    1. Split loaded documents into smaller chunks.
    2. Convert chunks into vector embeddings.
    3. Store embeddings in a FAISS vector index.
    4. Retrieve relevant chunks for a user question.

This version does not use an LLM. It performs retrieval only.
"""

import logging
# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def create_vector_store(documents):
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

    # Convert text chunks into numerical vectors using a local
    # Sentence Transformer model from Hugging Face.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
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


def create_qa_chain(vector_store):
    """
    Create a local retrieval-only question answering chain.

    Args:
        vector_store:
            FAISS vector store created from embedded document chunks.

    Returns:
        A callable function that accepts a question and returns
        retrieved text chunks plus their source documents.

    Note:
        This function does not generate a new answer with an LLM.
        It only retrieves the most relevant chunks from the documents.
    """
    
    # Use cuda if torch.cuda is available otherwise use cpu
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert the vector store into a retriever object.
    # MMR is used to balance relevance with diversity, reducing
    # the chance of returning near-duplicate chunks.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        model_kwargs={"device": DEVICE},
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.8
        }
    )

    def qa_chain(question):
        """
        Retrieve relevant document chunks for a user question.
        """

        source_documents = retriever.invoke(question)

        # Combine retrieved chunks into a readable response.
        # Since there is no LLM, the "answer" is the raw retrieved text.
        answer = "\n\n".join(
            doc.page_content for doc in source_documents
        )

        return {
            "result": answer,
            "source_documents": source_documents
        }

    return qa_chain
