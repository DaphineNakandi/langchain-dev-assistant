import streamlit as st
import os
from backend import get_llm_response
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangChain Assistant", page_icon="🦜", layout="wide")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ About this App")
    st.write("This is an AI-powered LangChain Developer Assistant.")
    st.write("It uses **RAG** to search local documentation and **Tool Calling** to calculate token costs and check the time.")
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🦜 LangChain Developer Assistant")

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error("⚠️ OPENROUTER_API_KEY not found.")
    st.stop() 

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🦜"
    with st.chat_message(message["role"], avatar=avatar):
        # If it's the assistant, show the tools it used
        if message["role"] == "assistant" and message.get("tools"):
            st.caption(f"🛠️ **Tools used:** {', '.join(message['tools'])}")
            
        st.write(message["content"])
        
        if message["role"] == "assistant" and "cost" in message:
            st.caption(f"💰 Cost: ${message['cost']:.6f} | 🪙 Tokens: {message['tokens']}")

# Chat Input
if user_query := st.chat_input("Ask a technical question (e.g., 'How do I split text?')"):

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_query)

    with st.chat_message("assistant", avatar="🦜"):
        with st.spinner("🧠 Agent is thinking and calling tools..."):
            result = get_llm_response(user_query)
            
            # Show tools if used
            if result.get("tools"):
                st.caption(f"🛠️ **Tools used:** {', '.join(result['tools'])}")
                
            st.write(result["response"])
            st.caption(f"💰 Cost: ${result['cost']:.6f} | 🪙 Tokens: {result['tokens']}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["response"],
        "cost": result["cost"],
        "tokens": result["tokens"],
        "tools": result.get("tools", [])
    })