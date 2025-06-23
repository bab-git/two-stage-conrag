# helper_gui.py

import streamlit as st
import os
from typing import Dict, List, Any, Tuple, Optional

# logging configured in backend/settings.py
import logging
from backend.my_lib.qa_chains import QAchains
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv, find_dotenv

# importing OmegaConf for loading model configs
from omegaconf import OmegaConf


# Load secrets from .env
load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)


# ===============================
# PDF uploader
# ===============================
def pdf_uploader_ui() -> tuple[list[UploadedFile] | None, str | None]:
    """
    Display the PDF uploader UI block and return a list of UploadedFile objects
    when the user clicks “Submit PDFs”. Returns None otherwise.
    """
    st.header("1. Upload PDF Documents")
    uploaded = st.file_uploader(
        "Upload PDF files. Loading time depends on total file size.",
        type="pdf",
        accept_multiple_files=True,
    )
    # Create two columns for the buttons
    col1, col2, col3 = st.columns([1, 0.31, 1])  # Adjust ratios for spacing

    with col1:
        use_samples = st.button(
            "📚 Process Sample PDFs",
            help="Load a set of built-in sample PDFs for quick demo testing.",
            type="primary",
        )

    with col2:
        st.markdown("#### or")

    with col3:
        submit_uploaded = st.button(
            "📁 Process Uploaded PDFs",
            help="Use the PDF files you've uploaded above to build the demo.",
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
    st.header("2. Ask a question")

    # ADD DEBUG BUTTON
    # col1, col2 = st.columns([3, 1])
    # with col2:
    #     if st.button("🔄 Clear Model Cache"):
    #         st.cache_resource.clear()
    #         st.success("Model cache cleared!")
    #         st.rerun()

    # with col1:
    question = st.text_area(
        "Enter your question related to the uploaded documents:",
        value="""What are the expectations for the Federal Reserve's interest rate cuts \
according to David Sekera, and how do these expectations relate to the \
upcoming Fed meetings and inflation data?""",
        height=80,  # Initial height (can be adjusted)
        # max_chars=1000,  # Optional: limit character count
        help="You can expand this field by dragging the bottom-right corner",
    )

    answer = None
    if st.button("Submit Question"):
        if question.strip():
            logger.info("Question submitted: %s", question)
            answer = process_question(question, qa_chains)

            # Display selected documents after processing
            # if hasattr(qa_chains, 'selected_documents') and qa_chains.selected_documents:
            # display_selected_documents(qa_chains.selected_documents, qa_chains.drs_scores)

        else:
            st.error("Please enter a question.")
            logger.warning("Empty question submitted.")
    return question.strip(), answer


def display_selected_documents(
    selected_documents: list[str], drs_scores: dict[str, float]
) -> None:
    """
    Display the selected documents with DRS scores in a simple expandable list.

    Args:
        selected_documents: List of selected document names
        drs_scores: Dictionary of document names and their normalized DRS scores
    """
    if not selected_documents:
        return

    with st.expander(
        f"📄 Selected Documents ({len(selected_documents)})", expanded=False
    ):
        for i, doc_name in enumerate(selected_documents, 1):
            score = drs_scores.get(doc_name, 0.0)
            st.write(f"{i}. {doc_name} - DRS: {score:.3f}")


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
        answer = None
        with st.spinner("Shortening question..."):
            if st.session_state.verbose:
                st.info(f"question: {question}")
            qa_chains.shorten_question(question)
            logger.info("========= Question shortened successfully.")

        with st.spinner("Searching for relevant documents..."):
            qa_chains.retrieve_context()
            logger.info("========= Context retrieved successfully.")

            # # Display selected documents after retrieval
            # if hasattr(qa_chains, 'selected_documents') and qa_chains.selected_documents:
            #     display_selected_documents(qa_chains.selected_documents, qa_chains.drs_scores)

        with st.spinner("Generating answer..."):
            answer = qa_chains.generate_answer()
            logger.info("========= Answer generated successfully.")
            if st.session_state.verbose:
                st.info(f"answer: {answer}")
                print(f"answer: {answer}")

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
    if answer:
        st.header("3. 💡 Answer")
        st.write(answer)

    with st.sidebar:
        st.header("📋 Results")

        if answer:
            st.subheader("💡 Current Answer")
            # Use smaller text area or expander
            # st.info(answer)
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f8ff;
                    border: 1px solid #d1ecf1;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    max-height: 300px;
                    overflow-y: auto;
                ">
                    <p style="
                        margin: 0;
                        line-height: 1.6;
                        color: #333;
                        font-size: 14px;
                    ">{answer}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            logger.info("Displayed current answer.")
            # with st.expander("View Full Answer", expanded=False):
            #     st.write(answer)

        if qa_history:
            st.subheader("📚 Q&A History")
            # Use an expander to make it collapsible if the history gets long
            with st.expander("View History", expanded=True):
                for idx, (q, a, model) in enumerate(qa_history, 1):
                    st.markdown(f"**Model:** `{model}`")
                    st.markdown(f"**Q{idx}:** {q}")
                    st.markdown(f"**A{idx}:** {a}")
                    st.markdown("---")
            logger.info("Displayed Q&A history.")


# ===============================
# Model Selection
# ===============================
def get_deployment_mode() -> str:
    """Get deployment mode from environment or Streamlit secrets."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, "secrets") and "DEPLOYMENT_MODE" in st.secrets:
            return st.secrets["DEPLOYMENT_MODE"]
    except (AttributeError, KeyError):
        pass

    # Fallback to environment variable (for local development)
    return os.getenv("DEPLOYMENT_MODE", "local")


def get_in_memory_mode() -> bool:
    """Get in-memory mode from environment or Streamlit secrets."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, "secrets") and "IN_MEMORY" in st.secrets:
            return st.secrets["IN_MEMORY"].lower() == "true"
    except (AttributeError, KeyError):
        pass

    # Fallback to environment variable (for local development)
    return os.getenv("IN_MEMORY", "false").lower() == "true"


def load_model_configs(config: OmegaConf) -> Dict[str, List[Dict]]:
    """Load available models based on deployment mode."""
    deployment_mode = get_deployment_mode()

    if deployment_mode == "local":
        return {
            "[Paid] OpenAI Models": config.models.local.openai,
            "[Free] Local Models": config.models.local.local_llama,
        }
    else:  # cloud
        return {
            "[Paid] OpenAI Models": config.models.cloud.openai,
            "[Free] Groq Models": config.models.cloud.groq,
        }


def check_api_key_availability(model_config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if required API key is available for the selected model.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Tuple of (is_available, message)
    """
    provider = model_config.get("provider")
    requires_key = model_config.get("requires_key", False)

    if not requires_key:
        return True, "No API key required"

    if provider == "openai":
        # For OpenAI, we'll handle key input in the UI
        return True, "OpenAI key will be requested"

    elif provider == "groq":
        # Check if Groq key is available in secrets or environment
        groq_key = ""

        # Try Streamlit secrets first (for cloud deployment)
        try:
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                groq_key = st.secrets["GROQ_API_KEY"]
        except (AttributeError, KeyError):
            pass

        # Fall back to environment variable
        if not groq_key:
            groq_key = os.getenv("GROQ_API_KEY", "")

        if groq_key and groq_key != "your-groq-api-key-here":
            return True, "Groq API key loaded from secrets"
        else:
            return False, "Groq API key not found in environment variables or secrets"

    return False, f"Unknown provider: {provider}"


def get_openai_key():
    """
    1. Reads OPENAI_API_KEY from .env in the repo root (without setting os.environ).
    2. If missing or equal to "dummy", prompts the user via a Streamlit text_input.
    3. Returns the key (may be empty if the user hasn’t typed it yet).
    """

    # Check environment first
    openai_key = os.getenv("OPENAI_API_KEY", "dummy").strip()

    # if it’s not set or is literally "dummy", ask the user
    if not openai_key or openai_key.lower() == "dummy":
        st.sidebar.header("🔑 OpenAI API Key Required")
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key.",
            type="password",
            help="This key will be stored in your environment—just for this session.",
        ).strip()

        if not api_key:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key to continue.")
            return None

        return api_key

    return openai_key


def select_model_ui(config: OmegaConf) -> Optional[Dict[str, Any]]:
    """
    Display model selection UI with deployment-aware options and API key management.

    Args:
        config: Configuration object containing model definitions

    Returns:
        Dict containing selected model configuration or None if not ready
    """
    with st.sidebar:
        st.header("🤖 Model Selection")

        # Load available models
        model_configs = load_model_configs(config)

        # Create flat list of models with category info
        model_options = ["Select a model..."]  # Add default option
        model_lookup = {}

        for category, models in model_configs.items():
            for model in models:
                display_name = f"{category} → {model['name']}"
                model_options.append(display_name)
                model_lookup[display_name] = {**model, "category": category}

        # Model selection dropdown
        selected_display_name = st.selectbox(
            "Choose your model:",
            options=model_options,
            index=0,  # Default to first option ("Select a model...")
            help="Select the model you want to use for question answering.",
            key="model_selector",  # Add unique key
        )

        # Return None if default option is selected
        if not selected_display_name or selected_display_name == "Select a model...":
            st.info("👆 Please select a model to continue")
            return None

        selected_model = model_lookup[selected_display_name]

        # Display model info
        with st.expander("ℹ️ Model Information", expanded=False):
            st.write(f"**Name:** {selected_model['name']}")
            st.write(f"**Provider:** {selected_model['provider'].title()}")
            st.write(f"**Model ID:** {selected_model['model_id']}")
            st.write(
                f"**Requires API Key:** {'Yes' if selected_model['requires_key'] else 'No'}"
            )

        # Check API key availability
        key_available, key_message = check_api_key_availability(selected_model)

        if not key_available:
            st.error(f"❌ **API Key Missing:** {key_message}")
            st.info(
                "💡 Please add the required API key to your environment variables and restart the app."
            )
            st.stop()

        # Handle API key input for OpenAI models
        api_key = None
        if selected_model["provider"] == "openai" and selected_model["requires_key"]:
            api_key = get_openai_key()
            if not api_key:
                st.stop()

        # Store API key in model config
        if api_key:
            selected_model["api_key"] = api_key

        # Success message
        provider_emoji = {"openai": "🤖", "groq": "⚡", "llama_cpp": "🦙"}.get(
            selected_model["provider"], "🔧"
        )

        st.success(f"{provider_emoji} **Selected:** {selected_model['name']}")

        return selected_model


# def display_loading_local_model():
#     st.sidebar.info("Loading local LLaMA model… this may take a moment.")
