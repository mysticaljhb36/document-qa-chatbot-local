"""
Streamlit application for the local document intelligence assistant.

This app loads documents from the data folder, builds a local FAISS
retrieval system, and provides a chat-style interface for asking
questions about the documents.

This version performs semantic retrieval only. It does not yet use
an LLM to generate final answers.
"""

import os

# Remove certificate bundle overrides that can cause Hugging Face
# model downloads to fail in some local/conda environments.
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import logging
# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

import streamlit as st

from src.paths import DATA_DIR
from src.document_loader import load_documents_from_folder
from src.rag_pipeline_local import create_vector_store, create_qa_chain
# Run loggers.py to initiate logging pipeline
from src.logger import loggers

# Controls the browser tab title and page layout.
st.set_page_config(
    page_title="Document Intelligence Assistant",
    layout="wide"
)

st.title("Document Intelligence Assistant")

st.caption(
    "Semantic document search using Sentence Transformers and FAISS "
    "created by Daniel Okereke"
)

# Streamlit reruns the script whenever the user interacts with the app. 
# Without @st.cache_resource caching, the documents and embeddings may be  
# rebuilt every time a user asks a question.
@st.cache_resource
def load_qa_system():
    """
    Load documents and initialise the local retrieval system.

    Streamlit caches this function so the documents, embeddings and
    FAISS vector store are not rebuilt every time the user asks a
    new question.
    """

    documents = load_documents_from_folder(DATA_DIR)

    vector_store = create_vector_store(documents)

    qa_chain = create_qa_chain(vector_store)

    return qa_chain


if __name__ == "__main__":
    
    logger.info(
    "Document Intelligence Assistant started"
    )

    try:

        qa_chain = load_qa_system()
    
    except Exception as error:
    
        logger.exception(
            f"Failed to initialise QA system: {error}"
        )
    
        st.error(
            "Application failed to initialise."
        )
    
        st.stop()

    # Store chat history so previous messages remain visible
    # after Streamlit reruns the script.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay existing messages on every rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # Creates the chat box at the bottom of the app.
    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        response = qa_chain(question)
        answer = response["result"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)