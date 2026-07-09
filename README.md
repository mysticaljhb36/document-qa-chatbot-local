# Document QA Chatbot (Local RAG Assistant)

A fully local Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, FAISS, Hugging Face Sentence Transformers and Ollama.

The application allows users to query PDF, DOCX and TXT documents through a chat-style interface. Documents are embedded locally using Sentence Transformers, indexed with FAISS, and relevant document content is retrieved before a local Large Language Model (LLM) generates a response.

No external vector database or cloud-based LLM is required.

---

## Features

- Load PDF, DOCX and TXT documents
- Automatic document chunking
- Semantic embeddings using Sentence Transformers
- Local FAISS vector database
- Local Ollama LLM integration
- Retrieval-Augmented Generation (RAG)
- Streamlit chat interface
- Conversation history
- GPU acceleration (CUDA) support
- Centralised logging
- Modular project structure
- Local-first architecture

---

## Project Structure

```text
document_qa_chatbot_local/

│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│
├── logs/
│   ├── .gitkeep
│   ├── logger.py
│   └── pipeline.log
│
└── src/
    ├── __init__.py
    ├── paths.py
    ├── document_loader.py
    ├── rag_pipeline_local.py
```

---

## How It Works

```text
Documents
    ↓
Document Loader
    ↓
Text Splitter
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Relevant Document Chunks
    ↓
Ollama (Llama 3.1)
    ↓
Generated Answer
```

The application uses Retrieval-Augmented Generation (RAG):

1. Documents are loaded and split into chunks.
2. Chunks are converted into embeddings.
3. Embeddings are stored in a local FAISS vector database.
4. Relevant chunks are retrieved for a user question.
5. Retrieved content is passed to a local Ollama LLM.
6. The LLM generates an answer grounded in the retrieved document information.

---

## Supported File Types

| Format | Supported |
|----------|----------|
| PDF | ✓ |
| DOCX | ✓ |
| TXT | ✓ |

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>

cd document_qa_chatbot_local
```

---

### 2. Create Virtual Environment

```bash
conda create -n rag-env python=3.11

conda activate rag-env
```

---

### 3. Install PyTorch with CUDA Support

This project supports NVIDIA GPU acceleration.

Check your CUDA version:

```bash
nvidia-smi
```

Install the appropriate PyTorch build:

https://pytorch.org/get-started/locally/

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output:

```text
True

NVIDIA GeForce RTX XXXX
```

---

### 4. Install Ollama

Download and install Ollama:

https://ollama.com

Pull the Llama 3.1 model:

```bash
ollama pull llama3.1
```

Verify Ollama is running:

```bash
ollama list
```

---

### 5. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch Streamlit:

```bash
streamlit run app.py
```

Application will open in your browser:

```text
http://localhost:8501
```

---

## Logging

Logs are written to:

```text
logs/pipeline.log
```

The logging system captures:

- Application startup
- Document loading
- Chunk creation
- Embedding generation
- FAISS vector store creation
- Retrieval operations
- Ollama answer generation
- Errors and exceptions

---

## Adding Documents

Place supported documents inside:

```text
data/
```

Example:

```text
data/

├── company_policy.pdf
├── user_guide.docx
└── notes.txt
```

The application automatically indexes all supported files.

---

## Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- Llama 3.1
- Hugging Face Sentence Transformers
- PyTorch
- CUDA
- Pathlib
- Logging

---

## Example Questions

```text
What does the company policy say about annual leave?
```

```text
Summarise the key points in this document.
```

```text
What are the responsibilities of the data protection officer?
```

```text
What is the process for reporting incidents?
```

```text
What information does this document contain about employee benefits?
```

---

## Future Enhancements

- Persistent FAISS storage
- Hybrid search (BM25 + Semantic Search)
- Document upload via Streamlit UI
- Separate document collections and workspaces
- Enhanced source citations with page previews and confidence scores
- Streaming LLM responses
- Conversation memory and contextual follow-up questions
- Retrieval reranking using Cross-Encoders
- RAG evaluation framework (RAGAS / DeepEval)
- User authentication and access control
- Docker containerisation
- Azure deployment pipeline
- Automated document ingestion and indexing

---

## Author

**Daniel Okereke**

Built as a learning and portfolio project demonstrating:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Natural Language Processing (NLP)
- Embeddings
- Information Retrieval
- Local LLM Deployment
- MLOps Best Practices
- Software Engineering Principles