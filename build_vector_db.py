import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load API key
load_dotenv()

def build_db():
    print("Loading document...")
    loader = TextLoader("data/langchain_basics.txt")
    docs = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print(f"Created {len(splits)} chunks.")
    
    # 2. Create embeddings using OpenRouter (mapped to OpenAI's embedding model)
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    # 3. Store in ChromaDB
    print("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db" # This creates a local folder
    )
    
    print("Database built successfully! Saved to ./chroma_db")

if __name__ == "__main__":
    build_db()