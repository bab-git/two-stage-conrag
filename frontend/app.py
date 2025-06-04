import os
import sys
import streamlit as st
from omegaconf import OmegaConf
# from dotenv import load_dotenv, find_dotenv
import getpass
# from dotenv import load_dotenv

# load_dotenv()

def initialize_debug_flag():
    st.session_state.debug = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Call it immediately — top of main app logic
if "debug" not in st.session_state:
    initialize_debug_flag()

# Ensure the root directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.my_lib.pdf_manager import PDFManager
from backend.my_lib.retrievers import Retrievers
from backend.my_lib.qa_chains import QAchains

from helper_gui import pdf_uploader_ui, question_input_output_ui

# Ensure environment variables are loaded and validated only once
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
    else:
        print(f"{var} is loaded")

# Ensure OPENAI_API_KEY is set
if 'env_loaded' not in st.session_state:
    _set_if_undefined("OPENAI_API_KEY")    
    st.session_state.env_loaded = True

def initialize_session_state():
    """
    Initialize necessary session state variables for Streamlit.
    """
    if 'retriever_large' not in st.session_state:
        st.session_state.retriever_large = None
    if 'retriever_small' not in st.session_state:
        st.session_state.retriever_small = None
    if 'pdf_manager' not in st.session_state:
        st.session_state.pdf_manager = None
    if 'answer' not in st.session_state:
        st.session_state.answer = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'debug' not in st.session_state:
        st.session_state.debug = False  # Default to False

def process_pdfs(pdf_path, config):
    """
    Process the uploaded PDF documents: load, chunk, and create a vector store.

    Args:
        pdf_path (str): Path to the folder containing PDF files.
        config (OmegaConf): Configuration object.
    """
    pdf_manager = PDFManager(pdf_path, config)

    with st.spinner("Loading PDFs..."):
        pdf_manager.load_pdfs()

    with st.spinner("Chunking documents..."):
        pdf_manager.chunk_documents()

    with st.spinner("Creating vector store..."):
        pdf_manager.create_vectorstore()

    retrievers = Retrievers(pdf_manager, config)

    with st.spinner("Creating retrievers..."):
        retrievers.setup_retrievers()

    # Update session state
    st.session_state.pdf_manager = pdf_manager
    st.session_state.retrievers = retrievers
    st.session_state.retriever_small = retrievers.retriever_small
    if st.session_state.get("retriever_small") is not None:
        st.success("PDFs and vector store processed successfully!")


def main():
    """
    The main function to define the Streamlit application for PDF-based Question Answering.
    """
    # Load configuration using OmegaConf
    config = OmegaConf.load("configs/config.yaml")
    # print(config)

    st.title("Two-Stage RAG System for PDF Question Answering")
    st.subheader("Fast yet Precise Document Retrieval and Question Answering")
    st.write("Upload a folder of PDF documents and ask questions to extract relevant information using the two-stage pipeline.")

    # Initialize session state variables
    initialize_session_state()
      
    # Show the PDF Upload Section
    pdf_path = pdf_uploader_ui()
    
    # Create the vector store and retrievers only if the pdfs are uploaded successfully
    if st.session_state.pdf_path is not None: 
        if st.session_state.debug:            
            st.write('pdfs path:', st.session_state.pdf_path)        
        if st.session_state.get("retriever_small") is None:
            process_pdfs(st.session_state.pdf_path, config)
        else:
            st.write('Vector store is already built - you can proceed to ask your question')
   
    # Show the Question Section only if the retriever is successfully created
    if st.session_state.get("retriever_small") is not None:
        st.session_state.qa_chain = QAchains(st.session_state.retrievers, config)  # Create QAchain
        question_input_output_ui(config)

if __name__ == "__main__":
    main()
