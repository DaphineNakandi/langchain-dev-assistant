import os
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_token_cost(num_tokens: int) -> str:
    """Calculates the cost in USD for a given number of tokens for GPT-4o-mini."""
    cost = (num_tokens / 1_000_000) * 0.15
    return f"Estimated cost for {num_tokens} tokens: ${cost:.6f}"

@tool
def search_langchain_docs(query: str) -> str:
    """Searches the local LangChain documentation for answers about text splitters, messages, and agents."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[get_current_time, calculate_token_cost, search_langchain_docs],
    system_prompt="You are a helpful LangChain developer assistant. Use the tools provided to answer questions."
)

# --- SECURITY: Prompt Injection Guardrail ---
def is_prompt_injection(text: str) -> bool:
    """A simple deterministic guardrail to catch obvious prompt injection attempts."""
    malicious_phrases = [
        "ignore previous instructions", 
        "ignore all previous", 
        "system prompt", 
        "you are now a",
        "forget your instructions"
    ]
    text_lower = text.lower()
    for phrase in malicious_phrases:
        if phrase in text_lower:
            return True
    return False

def get_llm_response(user_query: str) -> dict:
    # 1. Check for Prompt Injection
    if is_prompt_injection(user_query):
        return {
            "response": "⚠️ Security Alert: Your input was flagged as a potential prompt injection and was blocked.",
            "tokens": 0,
            "cost": 0.0,
            "tools": []
        }

    # 2. Invoke the agent
    response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    
    # 3. Extract text, token usage, and tools used
    final_message = response["messages"][-1].content
    total_tokens = 0
    tools_used = []

    for message in response["messages"]:
        # Get tokens
        if hasattr(message, "response_metadata") and "token_usage" in message.response_metadata:
            total_tokens += message.response_metadata["token_usage"].get("total_tokens", 0)
        
        # Get tool names
        if message.type == "tool":
            for prev_msg in response["messages"]:
                if hasattr(prev_msg, "tool_calls") and prev_msg.tool_calls:
                    for tc in prev_msg.tool_calls:
                        if tc["id"] == message.tool_call_id:
                            tools_used.append(tc["name"])
                            
    cost = (total_tokens / 1_000_000) * 0.15
    
    return {
        "response": final_message,
        "tokens": total_tokens,
        "cost": cost,
        "tools": list(set(tools_used)) # Return unique tool names
    }
