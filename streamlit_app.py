import os

import streamlit as st
from dotenv import load_dotenv

from backend import get_llm_response

load_dotenv()

st.set_page_config(
    page_title="LangChain Developer Assistant",
    page_icon=":material/terminal:",
    layout="wide",
)

# SIDEBAR
with st.sidebar:
    st.header("About this App")
    st.write(
        "This is an AI-powered LangChain Developer Assistant. "
        "It uses **RAG** to search local documentation and **Tool Calling** "
        "to calculate token costs and check the time."
    )
    st.divider()
    if st.button("Clear Chat History", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()

st.title("LangChain Developer Assistant")

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error(
        "OPENROUTER_API_KEY not found in environment variables.",
        icon=":material/error:"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    avatar = (
        ":material/person:" if message["role"] == "user" else ":material/smart_toy:"
    )
    with st.chat_message(message["role"], avatar=avatar):
        # If it's the assistant, show the tools it used
        if message["role"] == "assistant" and message.get("tools"):
            st.caption(f"**Tools used:** {', '.join(message['tools'])}")

        st.write(message["content"])

        if message["role"] == "assistant" and "cost" in message:
            st.caption(
                f"**Cost:** ${message['cost']:.6f} | **Tokens:** {message['tokens']}"
            )

# Chat Input
if user_query := st.chat_input("Ask a technical question (e.g., 'How do I split text?')"):

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=":material/person:"):
        st.write(user_query)

        with (
        st.chat_message("assistant", avatar=":material/smart_toy:"),
        st.spinner("Agent is thinking and calling tools..."),
    ):
         result = get_llm_response(user_query)

        # Show tools if used
        if result.get("tools"):
            st.caption(f"**Tools used:** {', '.join(result['tools'])}")

        st.write(result["response"])
        st.caption(
            f"**Cost:** ${result['cost']:.6f} | **Tokens:** {result['tokens']}"
        )
    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "cost": result["cost"],
            "tokens": result["tokens"],
            "tools": result.get("tools", []),
        }
    )
