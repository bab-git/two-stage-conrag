import os
import streamlit as st
from omegaconf import OmegaConf
from dotenv import load_dotenv, find_dotenv
import getpass

# Load environment variables from .env file
load_dotenv(find_dotenv())

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
    else:
        print(f"{var} already defined")

# Ensure OPENAI_API_KEY is set
_set_if_undefined("OPENAI_API_KEY")

from helper_functions import PDFManager, Retrievers, QAchains

def initialize_session_state():
    """
    Initialize necessary session state variables for Streamlit.
    """
    if 'retriever_large' not in st.session_state:
        st.session_state.retriever_large = None
    if 'pdf_manager' not in st.session_state:
        st.session_state.pdf_manager = None
    if 'answer' not in st.session_state:
        st.session_state.answer = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

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
    if st.session_state.retriever_small:
        st.success("PDFs and vector store processed successfully!")

def process_question(question, qa_chains):
    """
    Process the user's question by shortening it, retrieving relevant chunks, ranking them, and generating an answer.

    Args:
        question (str): The user's question.
        qa_chains (QAchains): An instance of the QAchains class.
    """
    try:
        with st.spinner("Shortening question..."):
            qa_chains.shorten_question(question)

        with st.spinner("Searching for relevant documents..."):
            qa_chains.retrieve_context()
            # qa_chains.rank_chunks()

        with st.spinner("Generating answer..."):
            answer = qa_chains.generate_answer()

        # Update session state
        st.session_state.answer = answer
        st.session_state.qa_history.append((question, answer))
    except Exception as e:
        st.error(f"An error occurred while processing the question: {e}")

def handle_question_section(config):
    """
    Handle the user question input and display the answer.

    Args:
        config (OmegaConf): Configuration object.
    """
    question = st.text_input(
        "Enter your question related to the uploaded documents:",
        value='''According to Morningstar's senior US economist Preston Caldwell, why is the 
Federal Reserve's forecast of a single interest rate cut in 2024 considered
too cautious, and what alternative scenario does Caldwell anticipate based 
on inflation trends?'''
    )

    qa_chains = QAchains(st.session_state.retrievers, config)

    if st.button("Submit Question"):
        if question:
            process_question(question, qa_chains)
        else:
            st.error("Please enter a question.")

    # Display the answer
    if st.session_state.get('answer'):
        st.subheader("Answer")
        st.text_area("Your Answer:", value=st.session_state.answer, height=200)

    # Display Q&A History
    if st.session_state.get('qa_history'):
        st.header("Q&A History")
        for idx, (q, a) in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Q{idx}:** {q}")
            st.markdown(f"**A{idx}:** {a}")
            st.markdown("---")

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

    # PDF Upload Section
    st.header("1. Upload PDF Documents")
    pdf_path = st.text_input(
        "Enter the path to the folder containing your PDF files:",
        value="data/sample_pdfs/"
    )

    if st.button("Submit PDFs"):
        if pdf_path and os.path.isdir(pdf_path):
            process_pdfs(pdf_path, config)
        else:
            st.error("Please enter a valid directory path containing PDF files.")

    # Question Section
    if st.session_state.get('retriever_small'):
        st.header("2. Ask a Question")
        handle_question_section(config)

if __name__ == "__main__":
    main()
