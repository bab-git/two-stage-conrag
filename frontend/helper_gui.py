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
        if pdf_path and os.path.isdir(pdf_path):
            # st.session_state.pdf_path = pdf_path
            return pdf_path
        else:
            st.error("Cannot find PDF files in the directory. Please select a directory with PDF files.")
            # else:
            #     st.error("Please enter a valid directory path.")
    # st.session_state.pdf_path = None
    return None

def question_input_output_ui(config, retrievers, qa_chains, qa_history):
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
    # qa_chains = st.session_state.qa_chain
    
    answer = None
    if st.button("Submit Question"):
        if question.strip():
            answer = process_question(question, qa_chains)
        else:
            st.error("Please enter a question.")
    return question.strip(), answer

def process_question(question, qa_chains):
    """
    Process the user's question by shortening it, retrieving relevant chunks, ranking them, and generating an answer.
    """
    if st.session_state.debug:
        answer = 'It\'s placeholder answer for debugging ' * 5
        # st.session_state.qa_history.append((question, st.session_state.answer))
        return answer
    try:
        with st.spinner("Shortening question..."):
            qa_chains.shorten_question(question)

        with st.spinner("Searching for relevant documents..."):
            qa_chains.retrieve_context()

        with st.spinner("Generating answer..."):
            answer = qa_chains.generate_answer()

    except Exception as e:
        st.error(f"An error occurred while processing the question: {e}")
        answer = None
    
    return answer



def display_results_ui(answer, qa_history):
    """
    Display the answer and Q&A history in the sidebar.
    Both answer and qa_history should be passed in from app.py's session state.
    """
    
    with st.sidebar:
        st.header("📋 Results")
        
        if answer:
            st.subheader("💡 Current Answer")
            # Use smaller text area or expander
            st.text_area("Your Answer:", value=answer, height=200, key="sidebar_answer")
            # with st.expander("View Full Answer", expanded=False):
            #     st.write(answer)

        if qa_history:
            st.subheader("📚 Q&A History")
            # Use an expander to make it collapsible if the history gets long
            with st.expander("View History", expanded=True):
                for idx, (q, a) in enumerate(qa_history, 1):
                    st.markdown(f"**Q{idx}:** {q}")
                    st.markdown(f"**A{idx}:** {a}")
                    # st.markdown(f"**A{idx}:** {a[:100]}{'...' if len(a) > 100 else ''}")  # Truncate long answers
                    st.markdown("---")