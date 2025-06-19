# helper_gui.py

import streamlit as st
import os
from backend.my_lib.qa_chains import QAchains
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv, find_dotenv

# Load secrets from .env
load_dotenv(find_dotenv(), override=True)

# logging configured in backend/settings.py
import logging

logger = logging.getLogger(__name__)

# ===============================
# PDF uploader
# ===============================
def pdf_uploader_ui() -> tuple[list[UploadedFile] | None, str | None]:
    """
    Display the PDF uploader UI block and return a list of UploadedFile objects
    when the user clicks “Submit PDFs”. Returns None otherwise.
    """
    st.header("2. Upload PDF Documents")
    uploaded = st.file_uploader(
        "Upload PDFs files or the folder containing PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    # Create two columns for the buttons
    col1, col2, col3 = st.columns([1, 0.31, 1])  # Adjust ratios for spacing

    with col1:
        submit_uploaded = st.button(
            "📁 Use Uploaded PDFs",
            help="Use the PDF files you've uploaded above to build the demo.",
            type="primary",
        )

    with col2:
        st.markdown("#### or")

    with col3:
        use_samples = st.button(
            "📚 Try Sample PDFs",
            help="Load a set of built-in sample PDFs for quick demo testing.",
            type="secondary",
        )

    # Handle uploaded PDFs
    if submit_uploaded:
        if uploaded:
            logger.info("User uploaded %d PDF(s)", len(uploaded))
            pdf_path = save_uploaded_pdfs(uploaded, "data/uploads")
            return uploaded, pdf_path
        else:
            st.error("Please upload at least one PDF.")
            logger.warning("Submit clicked with no PDFs uploaded")
            # return None, None

    # Handle sample PDFs
    if use_samples:
        sample_path = "data/sample_pdfs"
        if os.path.exists(sample_path) and os.path.isdir(sample_path):
            # Check if there are PDF files in the sample directory
            pdf_files = [
                f for f in os.listdir(sample_path) if f.lower().endswith(".pdf")
            ]
            if pdf_files:
                logger.info("Using sample PDFs from: %s", sample_path)
                st.success(
                    f"✅ Using {len(pdf_files)} sample PDF files from {sample_path}"
                )

                # Show which files will be used
                with st.expander("📋 Sample PDFs to be processed:", expanded=False):
                    for i, pdf_file in enumerate(pdf_files, 1):
                        st.write(f"{i}. {pdf_file}")

                return pdf_files, sample_path
            else:
                st.error(f"No PDF files found in {sample_path}")
                logger.warning("No PDF files in sample directory: %s", sample_path)
        else:
            st.error(f"Sample PDF directory not found: {sample_path}")
            logger.warning("Sample PDF directory does not exist: %s", sample_path)

    return None, None


# ===============================
# Save uploaded PDFs
# ===============================
def save_uploaded_pdfs(
    uploaded_files: list[UploadedFile], dest_folder: str, clear_existing: bool = True
) -> str:
    import shutil

    """
    Write the given UploadedFile objects to disk under dest_folder.
    If clear_existing is True, wipes the folder first.
    Returns the path to dest_folder once files are saved.
    """
    if clear_existing:
        shutil.rmtree(dest_folder, ignore_errors=True)
    os.makedirs(dest_folder, exist_ok=True)

    for f in uploaded_files:
        out_path = os.path.join(dest_folder, f.name)
        with open(out_path, "wb") as out:
            out.write(f.getbuffer())
    logger.info("Saved %d PDFs to %s", len(uploaded_files), dest_folder)
    return dest_folder


# def pdf_uploader_ui() -> str | None:
#     """
#     Display the PDF uploader UI block with input field and submit button.

#     This function renders a text input for users to enter a directory path containing
#     PDF files and a submit button to validate the path. It checks if the provided
#     path is a valid directory and displays appropriate error messages if not.

#     Returns:
#         str or None: The valid directory path if the submit button is clicked and
#                     the path is valid, otherwise None.

#     Side Effects:
#         - Displays Streamlit UI components (header, text_input, button)
#         - Shows error messages via st.error() for invalid paths
#     """
#     st.header("1. Upload PDF Documents")
#     pdf_path = st.text_input(
#         "Enter the path to the folder containing your PDF files:",
#         value="data/sample_pdfs/",
#     )

#     # Read from a local folder
#     if st.button("Submit PDFs"):
#         if pdf_path and os.path.isdir(pdf_path):
#             logger.info("PDF path submitted: %s", pdf_path)
#             return pdf_path
#         else:
#             st.error(
#                 "Cannot find PDF files in the directory. Please select a directory with PDF files."
#             )
#             logger.warning("Invalid PDF path submitted: %s", pdf_path)
#     return None


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
    st.header("3. Ask a question")
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



# ===============================
# Model Selection
# ===============================
def get_openai_key():
    """
    1. Reads OPENAI_API_KEY from .env in the repo root (without setting os.environ).
    2. If missing or equal to "dummy", prompts the user via a Streamlit text_input.
    3. Returns the key (may be empty if the user hasn’t typed it yet).
    """    

    api_key = os.getenv("OPENAI_API_KEY", "dummy").strip()

    # if it’s not set or is literally "dummy", ask the user
    if not api_key or api_key.lower() == "dummy":
        st.sidebar.header("OpenAI API Key") 
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key.",
            type="password",
            help="This key will be stored in your environment—just for this session.",
        ).strip()

    return api_key

def select_model_ui():
    st.header("1. Select LLM Model")
    model_choice = st.selectbox("Select LLM Backend", ["Local LLaMA", "OpenAI (GPT-4o-mini)"])
    return model_choice

# def display_loading_local_model():
#     st.sidebar.info("Loading local LLaMA model… this may take a moment.")