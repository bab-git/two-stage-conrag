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

from helper_gui import pdf_uploader_ui, question_input_output_ui, display_results_ui

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
    st.session_state.setdefault('env_loaded', False)
    st.session_state.setdefault('pdf_manager', None)
    st.session_state.setdefault('retrievers', None)
    st.session_state.setdefault('qa_chains', None)
    st.session_state.setdefault('answer', "")
    st.session_state.setdefault('qa_history', [])     
    st.session_state.setdefault('debug', False)   

@st.cache_resource
def vector_store_builder(pdf_path: str, _config):
    """
    Process the uploaded PDF documents: load, chunk, and create a vector store.

    Args:
        pdf_path (str): Path to the folder containing PDF files.
        config (OmegaConf): Configuration object.
    """
    # Step 1: Load and chunk
    pdf_manager = PDFManager(pdf_path, _config)        
    pdf_manager.load_pdfs()
    pdf_manager.chunk_documents()

    # Step 2: Create vector store
    pdf_manager.create_vectorstore()
    
    # Step 3: Create retrievers
    retrievers = Retrievers(pdf_manager, _config)
    retrievers.setup_retrievers()
    
    return pdf_manager, retrievers


def main():
    """
    The main function to define the Streamlit application for PDF-based Question Answering.
    """
    # Display the image at the top of the app    
    st.image("frontend/static/image.jpeg", use_container_width=True)

    # Load configuration using OmegaConf
    config = OmegaConf.load("configs/config.yaml")
    # print(config)

    st.title("Two-Stage RAG System for PDF Question Answering")
    st.subheader("Fast yet Precise Document Retrieval and Question Answering")
    st.write("Upload a folder of PDF documents and ask questions to extract relevant information using the two-stage pipeline.")
    if st.session_state.debug:
        st.warning('DEBUG MODE is ON')

    # Initialize session state variables
    initialize_session_state()
      
    # 1) PDF Upload and vector store creation
    pdf_path = pdf_uploader_ui()
    # Create vector store and retrievers only if the pdfs are uploaded successfully
    if pdf_path is not None: 
        if st.session_state.debug:            
            st.write('pdfs path:', pdf_path)        
              
        if st.session_state.get("retrievers") is not None:
            st.info('Vector store is already built - you can proceed to ask your question')
        pdf_manager, retrievers = vector_store_builder(pdf_path, config)
        st.session_state.pdf_manager = pdf_manager
        st.session_state.retrievers = retrievers
        # st.session_state.retriever_small = retrievers.retriever_small
        st.session_state.qa_chains = QAchains(retrievers, config)
        st.success("PDFs and vector store processed successfully!")        
   
    # 2) Question Section (only if retriever is successfully created)
    if st.session_state.get("retrievers") is not None:
        question, answer = question_input_output_ui(
            config,
            st.session_state.retrievers,
            st.session_state.qa_chains,
            st.session_state.qa_history,
        )

        if answer is not None:
            st.session_state.answer = answer
            st.session_state.qa_history.append((question, answer))
    
    # 3) Display answer & history
    display_results_ui(
        answer=st.session_state.answer,
        qa_history=st.session_state.qa_history,
    )

    
if __name__ == "__main__":
    main()
