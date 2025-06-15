# helper_gui.py

import streamlit as st
import os
from backend.my_lib.qa_chains import QAchains

# logging configured in backend/settings.py
import logging

logger = logging.getLogger(__name__)


# ===============================
# API Key Input
# ===============================
def api_key_input_ui() -> str | None:
    """
    Display API key input UI only if environment API key is invalid.
    First tries to use API key from environment, then prompts user if needed.
    
    Returns:
        str or None: The valid API key (from env or user input), otherwise None.
    
    Side Effects:
        - Stores user API key in session state for the current session
        - Displays informational message about key storage
    """
    # First check if we have a valid API key from environment
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    
    if env_api_key and env_api_key.startswith("sk-"):
        # Environment API key is valid, no need for user input
        st.success("✅ Using API Key from environment configuration")
        logger.info("Using valid API key from environment.")
        return env_api_key
    
    # Environment API key is invalid/missing, prompt user
    st.header("🔐 API Configuration Required")
    st.warning("⚠️ No valid OpenAI API key found in environment configuration.")
    
    # Check if user has already provided a key in this session
    if st.session_state.get("user_api_key"):
        st.success("✅ API Key configured for this session")
        st.info("Your key is not stored permanently. It's only used for your current session.")
        
        # Option to update/change the key
        if st.button("Update API Key"):
            st.session_state.user_api_key = None
            st.rerun()
        
        return st.session_state.user_api_key
    
    # API key input
    user_api_key = st.text_input(
        "🔐 Enter your OpenAI API Key", 
        type="password",
        help="Your API key will only be used for this session and is not stored permanently."
    )
    
    if user_api_key:
        if user_api_key.startswith("sk-"):
            st.session_state.user_api_key = user_api_key
            # Set environment variable for this session
            os.environ["OPENAI_API_KEY"] = user_api_key
            st.success("✅ API Key configured successfully!")
            st.info("Your key is not stored permanently. It's only used for your current session.")
            logger.info("User API key configured for session.")
            st.rerun()  # Refresh to show the success state
        else:
            st.error("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
            logger.warning("Invalid API key format provided.")
    
    st.info("💡 **Privacy Note:** Your API key is only stored in your browser session and is never saved to disk.")
    
    return st.session_state.get("user_api_key")

# ===============================
# PDF uploader
# ===============================
def pdf_uploader_ui() -> str | None:
    """
    Display the PDF uploader UI block with input field and submit button.

    This function renders a text input for users to enter a directory path containing
    PDF files and a submit button to validate the path. It checks if the provided
    path is a valid directory and displays appropriate error messages if not.

    Returns:
        str or None: The valid directory path if the submit button is clicked and
                    the path is valid, otherwise None.

    Side Effects:
        - Displays Streamlit UI components (header, text_input, button)
        - Shows error messages via st.error() for invalid paths
    """
    st.header("1. Upload PDF Documents")
    pdf_path = st.text_input(
        "Enter the path to the folder containing your PDF files:",
        value="data/sample_pdfs/",
    )

    if st.button("Submit PDFs"):
        if pdf_path and os.path.isdir(pdf_path):
            logger.info("PDF path submitted: %s", pdf_path)
            return pdf_path
        else:
            st.error(
                "Cannot find PDF files in the directory. Please select a directory with PDF files."
            )
            logger.warning("Invalid PDF path submitted: %s", pdf_path)
    return None


# ===============================
# Question input and output
# ===============================
def question_input_output_ui(qa_chains: QAchains) -> tuple[str, str | None]:
    """
    Handle user question input and process the question through the QA pipeline.

    This function displays a text input for users to enter questions, processes
    the question when submitted, and returns both the question and generated answer.
    It provides user feedback during processing and error handling.

    Args:
        config: Configuration object containing system settings and parameters.
        retrievers: Retrievers object for document retrieval operations.
        qa_chains (QAchains): Initialized QA chains object for question processing.
        qa_history (list): List of previous question-answer pairs for context.

    Returns:
        tuple: A tuple containing:
            - question (str): The trimmed user question
            - answer (str or None): The generated answer, or None if processing failed

    Side Effects:
        - Displays Streamlit UI components (text_input, button)
        - Shows processing spinners during question processing
        - Displays error messages for empty questions or processing failures
    """
    question = st.text_input(
        "Enter your question related to the uploaded documents:",
        value="""What are the expectations for the Federal Reserve's interest rate cuts 
according to David Sekera, and how do these expectations relate to the 
upcoming Fed meetings and inflation data?""",
    )

    answer = None
    if st.button("Submit Question"):
        if question.strip():
            logger.info("Question submitted: %s", question)
            answer = process_question(question, qa_chains)
        else:
            st.error("Please enter a question.")
            logger.warning("Empty question submitted.")
    return question.strip(), answer


# ===============================
# Process question
# ===============================
def process_question(question: str, qa_chains: QAchains) -> str | None:
    """
    Process a user question through the complete QA pipeline.

    This function handles the full question-answering workflow including question
    shortening, context retrieval, and answer generation. It provides visual
    feedback to users during each stage and handles any processing errors gracefully.

    Args:
        question (str): The user's question to be processed.
        qa_chains (QAchains): Initialized QA chains object containing the processing
                             pipeline including retrievers and language models.

    Returns:
        str or None: The generated answer string if processing succeeds,
                    None if an error occurs during processing.

    Side Effects:
        - Updates qa_chains internal state (shortened_question, retrieved_docs, etc.)
        - Displays processing spinners for each pipeline stage
        - Shows error messages via st.error() if processing fails
        - In debug mode, returns a placeholder answer without actual processing

    Raises:
        Exception: Any exception during QA processing is caught and displayed
                  to the user, with None returned as the result.
    """
    if st.session_state.debug:
        answer = "It's placeholder answer for debugging " * 5
        logger.debug("Debug mode active, returning placeholder answer.")
        return answer
    try:
        with st.spinner("Shortening question..."):
            qa_chains.shorten_question(question)
            logger.info("Question shortened successfully.")

        with st.spinner("Searching for relevant documents..."):
            qa_chains.retrieve_context()
            logger.info("Context retrieved successfully.")

        with st.spinner("Generating answer..."):
            answer = qa_chains.generate_answer()
            logger.info("Answer generated successfully.")

    except Exception as e:
        st.error(f"An error occurred while processing the question: {e}")
        logger.error("Error during question processing: %s", e)
        answer = None

    return answer


# ===============================
# Display results
# ===============================
def display_results_ui(
    answer: str | None, qa_history: list[tuple[str, str]] | None
) -> None:
    """
    Display the current answer and Q&A history in the Streamlit sidebar.

    This function renders a sidebar panel showing the most recent answer and
    a collapsible history of all previous question-answer pairs. It provides
    a clean, organized view of results that doesn't interfere with the main
    application workflow.

    Args:
        answer (str or None): The current answer to display. If None or empty,
                             no current answer section is shown.
        qa_history (list or None): List of tuples containing (question, answer) pairs
                                  representing the conversation history. If None or empty,
                                  no history section is shown.

    Side Effects:
        - Renders UI components in the Streamlit sidebar
        - Creates expandable sections for history viewing
        - Displays formatted question-answer pairs with separators

    Note:
        The function uses st.sidebar context manager to ensure all components
        are rendered in the sidebar rather than the main content area.
    """
    with st.sidebar:
        st.header("📋 Results")

        if answer:
            st.subheader("💡 Current Answer")
            # Use smaller text area or expander
            st.text_area("Your Answer:", value=answer, height=200, key="sidebar_answer")
            logger.info("Displayed current answer.")
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
            logger.info("Displayed Q&A history.")
