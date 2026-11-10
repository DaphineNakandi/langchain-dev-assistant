# LangChain Developer Assistant

An AI-powered domain-specific chatbot built with LangChain, Streamlit, and ChromaDB. This assistant is designed to help developers understand LangChain concepts using advanced RAG (Retrieval-Augmented Generation) and Tool Calling.

## Features

- **Advanced RAG:** Uses `RecursiveCharacterTextSplitter` to chunk local documentation and stores embeddings in a ChromaDB vector database.
- **Tool Calling:** The agent has access to 3 distinct tools:
  1. `get_current_time`: Returns the current date and time.
  2. `calculate_token_cost`: Estimates the USD cost of a given token count.
  3. `search_langchain_docs`: Searches the local vector database for LangChain documentation.
- **Security Guardrail:** Implements a deterministic prompt injection filter to block malicious inputs before they reach the LLM.
- **Cost & Token Tracking:** Every response displays the exact token usage and estimated cost.
- **RAGAs Evaluation:** Includes an `evaluate_rag.py` script to automatically score the faithfulness and relevancy of the RAG pipeline using the RAGAs framework.

## Architecture

- `streamlit_app.py`: The frontend chat interface.
- `backend.py`: The LangChain agent, tools, and security guardrail.
- `prompts.py`: Stores the system prompts (separated for clean code organization).
- `build_vector_db.py`: Script to chunk text, embed it, and populate ChromaDB.
- `evaluate_rag.py`: RAGAs evaluation script.
- `data/`: Contains the knowledge base (`langchain_basics.txt`).

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DaphineNakandi/langchain-dev-assistant.git
   cd langchain-dev-assistant

2. **Create and activate a virtual environment:**
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

3. **Install Dependencies:**
pip install streamlit langchain langchain-openai langchain-chroma langchain-community chromadb python-dotenv ragas datasets

4. **Set up your API key:**
OPENROUTER_API_KEY="your-openrouter-api-key"

5. **Build the vector database (Optional):**
   *Note: The `chroma_db` folder is already included in this repository, so you can skip this step.* 
   If you want to rebuild the database from scratch, run:
   ```bash
   python build_vector_db.py