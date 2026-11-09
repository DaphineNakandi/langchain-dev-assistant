# 🦜 LangChain Developer Assistant

An advanced RAG (Retrieval-Augmented Generation) chatbot designed to help developers navigate LangChain documentation. Built with Streamlit, LangChain, ChromaDB, and OpenRouter.

## 🌐 Live Demo
[Click here to view the live app](https://verbose-telegram-q7rwr457wrxj24vr5.github.dev/)


## ✨ Features
- **Advanced RAG:** Uses ChromaDB and `text-embedding-3-small` to retrieve relevant documentation chunks.
- **Tool Calling Agent:** Powered by LangChain's `create_agent`. The LLM dynamically decides when to use tools.
- **3 Custom Tools:**
  1. `search_langchain_docs`: Semantic search over the knowledge base.
  2. `get_current_time`: Returns the current server time.
  3. `calculate_token_cost`: Calculates the USD cost for a given number of tokens.
- **Security:** A deterministic prompt injection guardrail blocks malicious inputs before they reach the LLM.
- **Cost Tracking:** Displays the exact token usage and USD cost for every AI response.
- **RAG Evaluation:** Includes `evaluate_rag.py`, a RAGAs script to test faithfulness and answer relevancy.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone <>
   cd langchain-dev-assistant