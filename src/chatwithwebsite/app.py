import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# App Configuration
st.set_page_config(
    page_title="Chat With WebSites",
    page_icon="🤖"
)

st.title("Chat With WebSites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")


user_query = st.chat_input("Enter the Promt Here...")
if user_query is not None and user_query != "":

    with st.chat_message("AI"):
    st.write("Hello, How Can I Help You?")

    with st.chat_message("Human"):
    st.write("Hello, How Can I Help You?")


