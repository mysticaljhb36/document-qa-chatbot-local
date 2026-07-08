# =============================================================================
# Streamlit Application: Local Document Intelligence Assistant
#
# Loads and indexes documents using a cached FAISS vector store,
# accepts user questions through a chat interface, and generates
# answers using a local Ollama Large Language Model (LLM).
#
# Components:
#     - Streamlit (User Interface)
#     - Sentence Transformers (Embeddings)
#     - FAISS (Vector Retrieval)
#     - Ollama / Llama 3.1 (Answer Generation)
#
# Author: Daniel Okereke
# =============================================================================

"""
Streamlit application for the local document intelligence assistant.

This app loads documents from the data folder, builds a local FAISS
retrieval system, and uses Ollama to generate answers from the
retrieved document content.

Users can interact with the assistant through a chat-style interface
to ask questions about their documents.

Retrieval is performed locally using Sentence Transformers and FAISS,
while answer generation is handled by a local Ollama LLM.
"""

import os

# Remove certificate bundle overrides that can cause Hugging Face
# model downloads to fail in some local/conda environments.
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import logging

import streamlit as st

from src.paths import DATA_DIR
from src.document_loader import load_documents_from_folder
from src.rag_pipeline_local import (
    create_vector_store,
    generate_answer
)
# Run loggers.py to initiate logging pipeline
from src.logger import loggers

# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

# Controls the browser tab title and page layout.
st.set_page_config(
    page_title="Document Intelligence Assistant",
    layout="wide"
)

st.title("Document Intelligence Assistant")

st.caption(
    "Local Retrieval-Augmented Generation (RAG) assistant powered by "
    "Sentence Transformers, FAISS and Ollama • Created by Daniel Okereke"
)

# Initialise chat history once per user session.
# Session state allows conversations to persist across
# Streamlit script reruns.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit reruns the script whenever the user interacts with the app.
# Without @st.cache_resource caching, the documents and embeddings may be
# rebuilt every time a user asks a question.
@st.cache_resource
def load_vector_store():
    """
    Load documents and initialise the local retrieval system.

    Streamlit caches this function so the documents, embeddings and
    FAISS vector store are not rebuilt every time the user asks a
    new question.
    """

    # Load all supported documents from the data directory.
    documents = load_documents_from_folder(DATA_DIR)

    # Create embeddings and build the FAISS vector index.
    vector_store = create_vector_store(documents)

    return vector_store


if __name__ == "__main__":

    logger.info(
        "Document Intelligence Assistant started"
    )

    try:
        # Load the cached vector store. If this is the first run,
        # documents will be processed and indexed automatically.
        vector_store = load_vector_store()

    except Exception as error:

        logger.exception(
            f"Failed to initialise QA system: {error}"
        )

        st.error(
            "Application failed to initialise."
        )

        st.stop()

    # Display a ChatGPT-style input box for document questions.
    question = st.chat_input("Ask a question about the document...")

    if question:
        logger.info(f"Question received: {question}")

        # Retrieve relevant document chunks and generate
        # an answer using the local Ollama model.
        answer = generate_answer(
            question=question,
            vector_store=vector_store
        )

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

    # Display the conversation history.
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Assistant:** {chat['answer']}")
        st.divider()