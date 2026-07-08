"""
Standalone FAISS + Ollama RAG test.

This script tests the full local RAG flow without Streamlit:
documents -> chunks -> FAISS retriever -> Ollama answer.
"""
import os

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from langchain_ollama import ChatOllama

from src.paths import DATA_DIR
from src.document_loader import load_documents_from_folder
from src.rag_pipeline_local import create_vector_store


def main():
    question = "What is this document about?"

    print("Loading documents...")
    documents = load_documents_from_folder(DATA_DIR)

    print("Creating FAISS vector store...")
    vector_store = create_vector_store(documents)

    print("Creating retriever...")
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    print("Retrieving relevant chunks...")
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    llm = ChatOllama(
        model="llama3.1",
        temperature=0
    )

    prompt = f"""
You are a helpful document assistant.

Answer the question using only the context below.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:
"""

    print("Generating answer with Ollama...")
    response = llm.invoke(prompt)

    print("\nANSWER:")
    print(response.content)

    print("\nSOURCES:")
    for doc in retrieved_docs:
        print(doc.metadata.get("source", "Unknown source"))


if __name__ == "__main__":
    main()