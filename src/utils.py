import streamlit as st
import sys
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
import os
import base64

# Load the environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
key = os.getenv("OPENAI_API_KEY")

def get_vectorstore_from_url(url):
    """
    Retrieves text content from a given URL, creates vector embeddings,
    and stores those embeddings in a Chroma vector store.

    Args:
        url: The URL of the web page to process.

    Returns:
        A Chroma vector store containing the embeddings of the text chunks from the URL.
    """
    try:

        # 1. Load the text content from the URL:
        loader = WebBaseLoader(url)  # Initialize a WebBaseLoader object to fetch text from the URL
        document = loader.load()     # Load the text content into a document object

        # 2. Split the text into manageable chunks:
        text_splitter = RecursiveCharacterTextSplitter()  # Create a splitter for efficient chunking
        document_chunks = text_splitter.split_documents(document)  # Divide the text into smaller chunks

        # 3. Generate vector embeddings for each chunk:
        vector_store = Chroma.from_document(
            document_chunks, OpenAIEmbeddings()  # Create a Chroma vector store using OpenAI's embeddings
        )

        return vector_store  # Return the created vector store for further use
    
    except Exception as e:
        logging.info("Error Occured In utils.py in get_vectorstore_from_url")
        raise CustomException(e,sys)


def get_context_retriever_chain(vector_store):
    """
    (Potentially using an API) Creates a retriever chain that leverages a large language model
    (LLM) for context-aware information retrieval from a vector store.

    Args:
        vector_store: A Chroma vector store containing text embeddings.

    Returns:
        A retriever chain that combines an LLM-based contextualizer and the provided vector store retriever.

    Raises:
        ImportError: If the `ChatOpenAI` class or `ChatPromptTemplate` class from an unspecified library is not found.
    """
    try:
        llm = ChatOpenAI(
           # Initialize an LLM instance
            openai_api_key = key,
            model = "gpt-3.5-turbo",
            temperature=0.9
        )

         # Create a retriever from the vector store
        retriever = vector_store.as_retriever()

        # Define a prompt template using placeholders for conversation history and user input
        prompt = ChatPromptTemplate.from_messages(
            [MessagesPlaceholder(variable_name="chat_history"),
             ("user","{input}"),
             ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
             ]
        )
        retriever_chain = create_history_aware_retriever(
            llm,
            retriever,
            prompt
        )

        return retriever_chain

    except Exception as e:
        logging.info("Error Occured In utils.py in get_context_retriever_chain")
        raise CustomException(e,sys)
    

def get_conversational_rag_chain(retriever_chain):
    """
    Function to create a conversational retrieval-augmented generation (RAG) chain.

    Args:
    - retriever_chain (list): List of retriever instances forming a retriever chain.

    Returns:
    - rag_chain (list): List of components forming a RAG chain for conversational response generation.
    """

    try:
        llm = ChatOpenAI(
           # Initialize an LLM instance
            openai_api_key = key,
            model = "gpt-3.5-turbo",
            temperature=0.9
        )

         # Define a prompt template for the conversation
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ])
        
        # Create a chain of stuff documents using the llm and prompt
        stuff_documents_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        # Create the final retrieval-augmented generation (RAG) chain
        rag_chain = create_retrieval_chain(retriever_chain,
                                            stuff_documents_chain)

        return rag_chain
        
    except Exception as e:
        logging.info("Error Occured In utils.py in get_conversational_rag_chain")
        raise CustomException(e,sys)
    


def get_response(user_input):
    """
    Function to generate a response based on user input.

    Args:
    - user_input (str): The input provided by the user.

    Returns:
    - response (str): The generated response.
    """
    # Get the retriever chain using vector store from session state
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    
    # Get the conversational retrieval-augmented generation (RAG) chain using the retriever chain
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # Invoke the conversational RAG chain with chat history and user input
    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    # Return the generated answer from the response
    return response["answer"]


def set_background(image_file):
    try:
        """
        This function sets the background of a Streamlit app to an image specified by the given image file.

        Parameters:
        image_file (str): The path to the image file to be used as the background.

        Returns:
        None
        """
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        style = f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{b64_encoded});
             background-size: cover;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except Exception as e:
        logging.info("Error occurred while setuping background image:")
        raise Exception(e,sys)
