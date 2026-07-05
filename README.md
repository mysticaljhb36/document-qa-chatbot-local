\# Document QA Chatbot (Local RAG Assistant)



A local Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, FAISS, and Hugging Face embeddings.



The application allows users to upload and query documents locally without relying on external vector databases. Documents are indexed using FAISS and relevant context is retrieved before answering user questions.



\---



\## Features



\- Load PDF, DOCX and TXT documents

\- Automatic document chunking

\- Semantic embeddings using Sentence Transformers

\- Local FAISS vector database

\- Streamlit chat interface

\- GPU acceleration (CUDA) support

\- Centralised logging

\- Modular project structure

\- Local-first architecture



\---



\## Project Structure



```text

document\_qa\_chatbot\_local/

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

&#x20;   ├── \_\_init\_\_.py

&#x20;   ├── paths.py

&#x20;   ├── document\_loader.py

&#x20;   └── rag\_pipeline\_local.py

```



\---



\## How It Works



```text

Documents

&#x20;   ↓

Document Loader

&#x20;   ↓

Text Splitter

&#x20;   ↓

Embeddings

&#x20;   ↓

FAISS Vector Store

&#x20;   ↓

Retriever

&#x20;   ↓

User Question

&#x20;   ↓

Relevant Chunks Returned

```



Current implementation performs retrieval and returns the most relevant document chunks.



Future versions may integrate a local LLM to provide full Retrieval-Augmented Generation (RAG).



\---



\## Supported File Types



| Format | Supported |

|----------|----------|

| PDF | ✓ |

| DOCX | ✓ |

| TXT | ✓ |



\---



\## Installation



\### 1. Clone Repository



```bash

git clone <repository-url>

cd document\_qa\_chatbot\_local

```



\---



\### 2. Create Virtual Environment



```bash

conda create -n rag-env python=3.11

conda activate rag-env

```



\---



\### 3. Install PyTorch with CUDA Support



This project supports NVIDIA GPU acceleration.



Check your CUDA version:



```bash

nvidia-smi

```



Install the appropriate PyTorch build from:



https://pytorch.org/get-started/locally/



Example for CUDA 12.1:



```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```



Verify installation:



```python

import torch



print(torch.cuda.is\_available())

print(torch.cuda.get\_device\_name(0))

```



Expected output:



```text

True

NVIDIA GeForce RTX XXXX

```



\---



\### 4. Install Project Dependencies



```bash

pip install -r requirements.txt

```



\---



\## Running the Application



Launch Streamlit:



```bash

streamlit run app.py

```



Application will open in your browser:



```text

http://localhost:8501

```



\---



\## Logging



Logs are written to:



```text

logs/pipeline.log

```



The logging system captures:



\- Application startup

\- Document loading

\- Chunk creation

\- Embedding generation

\- Vector store creation

\- Retrieval operations

\- Errors and exceptions



\---



\## Adding Documents



Place supported documents inside:



```text

data/

```



Example:



```text

data/

├── company\_policy.pdf

├── user\_guide.docx

└── notes.txt

```



The application automatically indexes all supported files.



\---



\## Technologies Used



\- Python

\- Streamlit

\- LangChain

\- FAISS

\- Hugging Face Sentence Transformers

\- PyTorch

\- CUDA

\- Pathlib

\- Logging



\---



\## Example Questions



```text

What does the company policy say about annual leave?



Summarise the key points in this document.



What are the responsibilities of the data protection officer?



What is the process for reporting incidents?

```



\---



\## Future Enhancements



\- Local LLM integration (Mistral, Llama, Gemma, OpenAI)

\- Source citation display

\- Conversation memory

\- Multi-document collections

\- Persistent FAISS storage

\- Hybrid search

\- Document upload via UI



\---



\## Author



Daniel Okereke



Built as a learning and portfolio project demonstrating:



\- Retrieval-Augmented Generation (RAG)

\- Vector Databases

\- NLP

\- Embeddings

\- Information Retrieval

\- MLOps Best Practices

\- Software Engineering Principles

