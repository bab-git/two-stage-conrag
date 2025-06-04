# helper_gui.py

import streamlit as st
import os
from backend.my_lib.qa_chains import QAchains

def pdf_uploader_ui():
    """
    Display the PDF uploader UI block.
    """
    st.header("1. Upload PDF Documents")
    pdf_path = st.text_input(
        "Enter the path to the folder containing your PDF files:",
        value="data/sample_pdfs/"
    )

    if st.button("Submit PDFs"):
        if pdf_path:
            if os.path.isdir(pdf_path):
                st.session_state.pdf_path = pdf_path
                return pdf_path
            else:
                st.error("The directory is empty. Please select a directory with PDF files.")
        else:
            st.error("Please enter a valid directory path.")
    st.session_state.pdf_path = None
    return None

def question_input_output_ui(config):
    """
    Handle the user question input and display the answer.
    """
    question = st.text_input(
        "Enter your question related to the uploaded documents:",
        value='''What are the expectations for the Federal Reserve's interest rate cuts 
according to David Sekera, and how do these expectations relate to the 
upcoming Fed meetings and inflation data?'''
    )

    # qa_chains = QAchains(st.session_state.retrievers, config)
    qa_chains = st.session_state.qa_chain

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

def process_question(question, qa_chains):
    """
    Process the user's question by shortening it, retrieving relevant chunks, ranking them, and generating an answer.
    """
    if st.session_state.debug:
        st.session_state.answer = 'It\'s placeholder answer for debugging'
        st.session_state.qa_history.append((question, st.session_state.answer))
        return
    try:
        with st.spinner("Shortening question..."):
            qa_chains.shorten_question(question)

        with st.spinner("Searching for relevant documents..."):
            qa_chains.retrieve_context()

        with st.spinner("Generating answer..."):
            answer = qa_chains.generate_answer()

        # Update session state
        st.session_state.answer = answer
        st.session_state.qa_history.append((question, answer))
    except Exception as e:
        st.error(f"An error occurred while processing the question: {e}")