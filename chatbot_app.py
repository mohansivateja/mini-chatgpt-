import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load API key from .env

# Set Streamlit page config
st.set_page_config(page_title="ChatGPT Clone", page_icon="🤖")

# Initialize session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Chat model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
)

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Streamlit UI
st.title("🤖 ChatGPT Clone")
st.markdown("Chat with your AI assistant. It remembers your past messages!")

user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send") and user_input:
    response = conversation.predict(input=user_input)
    st.markdown(f"**AI:** {response}")

    # Show full memory history
    with st.expander("🧠 Conversation History"):
        st.markdown(st.session_state.memory.buffer)
