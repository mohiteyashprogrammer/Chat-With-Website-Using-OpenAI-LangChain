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
from src.utils import get_context_retriever_chain,get_response,get_conversational_rag_chain,get_vectorstore_from_url,set_background




# App Configuration
# Sets the title and icon for the Streamlit app
st.set_page_config(
    page_title="Chat With WebSites",
    page_icon="🤖"
)

# Creates a heading for the app
st.title("Chat With WebSites")

# Sidebar for URL input
with st.sidebar:
    st.header("Settings")
     # Text input field for the website URL
    website_url = st.text_input("Website URL")

# Error message if no URL is entered
if website_url is None or website_url == "":
    st.info("Please enter a website URL")

# Main logic
else:
    # Session state to store chat history and vector store
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content="Hello, I am a chat bot. How can I help you?"
            )
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

        # TODO: Error handling for potential issues with `get_vectorstore_from_url`

    
     # User input field for the chat prompt
    user_query = st.chat_input("Enter the Promt Here...")

    # Process user input if it's not empty
    if user_query is not None and user_query!= "":

        # Get the response from the chat model (replace `get_response` with your implementation)
        response = get_response(user_query)

        # Update the chat history with the new messages
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    
    # Display the chat history
    for message in st.session_state.chat_history:

        # Differentiate between AI and human messages using conditional formatting
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)