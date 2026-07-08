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
from langchain_ollama import ChatOllama

# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_vector_store(
    documents: list
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


def generate_answer(
    question: str,
    vector_store: FAISS
    ) -> str:
    
    """
    Generate an answer using retrieved document chunks and Ollama.

    Args:
        question:
            User question submitted through the Streamlit app.

        vector_store:
            FAISS vector store created from embedded document chunks.

    Returns:
        Generated answer from the local Ollama model.
    """

    # Convert the FAISS vector store into a retriever.
    # MMR helps return diverse but still relevant document chunks.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # Retrieve document chunks most relevant to the user's question.
    retrieved_docs = retriever.invoke(question)

    # Combine retrieved chunks into one document information block
    # that will be passed to the local LLM.
    document_information = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Initialise the local Ollama chat model only when an answer
    # is required. This avoids caching the LLM object in Streamlit.
    llm = ChatOllama(
        model="llama3.1",
        temperature=0
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

    return response.content