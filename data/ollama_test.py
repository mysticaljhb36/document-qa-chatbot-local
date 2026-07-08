"""
Standalone Ollama test.

This file tests whether the local Ollama model can respond
without touching the FAISS or Streamlit pipeline.
"""

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

response = llm.invoke(
    "Hello. Reply with one short sentence confirming you are working."
)

print(response.content)