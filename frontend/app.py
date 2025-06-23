import sys

try:
    # if pysqlite3 exists (i.e. you have installed it), load and swap it in
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    # Optionally log so you know it happened:
    print("🔄 Overriding stdlib sqlite3 with pysqlite3")
except ImportError:
    # no pysqlite3 installed → skip the swap (use system sqlite3)
    pass

import os
import shutil
import streamlit as st
from omegaconf import OmegaConf

# Conditional import for llama_cpp (only needed for local deployment)
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

# Ensure the root directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.my_lib.pdf_manager import PDFManager
from backend.my_lib.retrievers import Retrievers
from backend.my_lib.qa_chains import QAchains
from backend.settings import load_and_validate_env_secrets
from backend.my_lib.LLMManager import LLMManager
from helper_gui import (
    question_input_output_ui,
    display_results_ui,
    pdf_uploader_ui,
    select_model_ui,
    get_in_memory_mode,
    get_deployment_mode,
)

# logging from backend
import logging

logger = logging.getLogger(__name__)


def initialize_session_state() -> None:
    """
    Initialize necessary session state variables for Streamlit.
    """
    # Set 'debug' based on env var, but store it in session_state immediately
    st.session_state.setdefault(
        "debug", os.getenv("DEBUG_MODE", "false").lower() == "true"
    )
    st.session_state.setdefault("pdf_manager", None)
    st.session_state.setdefault("retrievers", None)
    st.session_state.setdefault("qa_chains", None)
    st.session_state.setdefault("answer", "")
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("selected_model", None)
    st.session_state.setdefault("llm_manager", None)
    st.session_state.setdefault("model_changed", False)
    st.session_state.setdefault("verbose", False)
    st.session_state.setdefault("api_key", None)
    logger.debug("Session state initialized.")


# Cache this resource so it's only loaded once per session
@st.cache_resource
def load_local_llama(repo_id: str, filename: str) -> Llama:
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is not available. This is expected for cloud deployment."
        )

    llama_instance = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        local_dir="models",
        n_ctx=10000,
        # n_batch=512,      # Add this
        verbose=False,
    )
    with st.sidebar:
        st.info("Local model loaded successfully.")
    return llama_instance


@st.cache_resource
def vector_store_builder(
    pdf_path: str, _config: OmegaConf, uploaded: list | None
) -> tuple[PDFManager, Retrievers]:
    """
    Process the uploaded PDF documents: load, chunk, and create a vector store.

    Args:
        pdf_path (str): Path to the folder containing PDF files.
        config (OmegaConf): Configuration object.
    """

    logger.info("Building vector store for PDFs at path: %s", pdf_path)

    # Step 1: Load and chunk
    pdf_manager = PDFManager(pdf_path, _config)
    pdf_manager.load_pdfs()
    pdf_manager.chunk_documents()

    # Step 2: Create vector store
    pdf_manager.create_vectorstore()

    # Step 3: Create retrievers
    retrievers = Retrievers(pdf_manager, _config)
    retrievers.setup_retrievers()

    logger.info("Vector store and retrievers created successfully.")
    return pdf_manager, retrievers


def main() -> None:
    """
    Entry point for the Streamlit application that drives the Two-Stage RAG PDF QA system.

    This function orchestrates the entire UI and backend workflow:
    1. Displays a header image and title/subtitle for the app.
    2. Loads configuration settings from the OmegaConf YAML file.
    3. Initializes all required Streamlit session state variables.
    4. Handles PDF upload and triggers vector store creation:
       - If PDFs are provided, invokes `vector_store_builder` to load, chunk, and index documents.
       - Stores the resulting PDFManager and Retrievers objects in session state.
       - Instantiates the QAchains object for downstream question answering.
    5. Renders the question‐input section once retrievers exist:
       - Passes the QAchains instance into `question_input_output_ui`.
       - Captures and stores the user’s question and generated answer in session state.
    6. Displays the latest answer and the Q&A history via `display_results_ui`.

    Session State Keys Used:
        - debug (bool): Toggles debug messages if True.
        - pdf_manager (PDFManager | None): Manages PDF loading and chunking.
        - retrievers (Retrievers | None): Encapsulates BM25 and semantic retrievers.
        - qa_chains (QAchains | None): Orchestrates question shortening, retrieval, and answer generation.
        - answer (str): The most recent answer generated by the pipeline.
        - qa_history (list[tuple[str, str]]): A chronological list of (question, answer) pairs.

    Returns:
        None
    """

    logger.info("Starting Streamlit app")

    # Display the image at the top of the app
    st.image("frontend/static/image.jpeg", use_container_width=True)

    # Load configuration using OmegaConf
    config = OmegaConf.load("configs/config.yaml")
    logger.info("Configuration loaded successfully.")
    # print(config)

    # ==============================
    # Constructing the Layout
    # ==============================
    st.title("Two-Stage RAG System for PDF Question Answering")
    # st.subheader("Fast yet Precise Document Retrieval and Question Answering")
    st.write(
        "Start by **selecting a model** (OpenAI or Open Models) from **left sidebar**, then **upload your PDF files**, and finally **ask questions** to extract insights using the two-stage retrieval system."
    )

    # sidebar
    st.sidebar.header("App Description")
    st.sidebar.write(
        "This application uses a two-stage retrieval-augmented generation (RAG) pipeline to efficiently extract information from PDF documents. "
        "It combines lexical retrieval (BM25) with semantic retrieval (vector embeddings) in two consecutive stages."
        "Upload your PDFs and ask questions to receive precise answers powered by either OpenAI's advanced models or free open-source models via Groq API (or llama-cpp-python in local deployment). "
    )
    # Show deployment mode
    deployment_mode = get_deployment_mode()
    deployment_emoji = "🏠" if deployment_mode == "local" else "☁️"
    st.sidebar.info(
        f"{deployment_emoji} **Deployment Mode:** {deployment_mode.title()}"
    )
    st.sidebar.info(
        f"📊 **Storage Mode:** {'In-Memory' if get_in_memory_mode() else 'Persistent'}"
    )

    # Initialize session state variables
    initialize_session_state()
    logger.info("Session state initialized successfully.")

    # Check verbose mode
    if config.settings.verbose:
        st.session_state.verbose = True
        st.warning("Verbose mode is enabled.")

    # Clear the vector store if needed
    if st.session_state.verbose:
        print(
            "vector_store_cleared:", st.session_state.get("vector_store_cleared", False)
        )
    if (
        not st.session_state.get("vector_store_cleared", False)
        and config.Vectorstore.clear_existing
    ):
        shutil.rmtree(config.Vectorstore.persist_directory, ignore_errors=True)
        # rebuild the vector store
        st.session_state.vector_store_cleared = True

    # Check debug mode
    if st.session_state.debug:
        st.warning("DEBUG MODE is ON")
        logger.debug("Debug mode is enabled.")

    # Loading existing environment secrets
    if not st.session_state.get("env_validated"):
        load_and_validate_env_secrets()
        st.session_state.env_validated = True
        logger.info("Environment secrets validated successfully.")

    # ==============================
    # Model Selection
    # ==============================
    selected_model = select_model_ui(config)

    if not selected_model:
        st.stop()

    # Check if model has changed
    model_changed = (
        st.session_state.selected_model is None
        or st.session_state.selected_model.get("model_id")
        != selected_model.get("model_id")
        or st.session_state.selected_model.get("provider")
        != selected_model.get("provider")
    )

    if model_changed:
        st.session_state.model_changed = True
        st.session_state.selected_model = selected_model
        # Clear existing LLM manager and QA chains when model changes
        st.session_state.llm_manager = None
        st.session_state.qa_chains = None

        if st.session_state.verbose:
            st.info(f"Model changed to: {selected_model['name']}")

    # Initialize LLM Manager based on selected model
    if st.session_state.llm_manager is None or model_changed:
        if selected_model["provider"] == "llama_cpp":
            # Load local LLaMA model
            with st.spinner("Loading local LLaMA model..."):
                repo_model = selected_model["model_id"]
                filename = selected_model["filename"]
                llama_instance = load_local_llama(repo_model, filename)

            llm_manager = LLMManager(selected_model)
            llm_manager.set_llama_instance(llama_instance)

        else:
            # OpenAI or Groq models
            api_key = selected_model.get("api_key")
            llm_manager = LLMManager(selected_model, api_key)

        st.session_state.llm_manager = llm_manager
        st.session_state.model_changed = False

    # Get the current llm_manager from session state
    llm_manager = st.session_state.llm_manager

    if st.session_state.verbose:
        print("====== Current llm choice and llm_manager:", selected_model, llm_manager)

    # ==============================
    # PDF Upload and vector store creation
    # ==============================
    uploaded, pdf_path = pdf_uploader_ui()
    if uploaded is not None:
        logger.info("PDF path provided: %s", pdf_path)
        if st.session_state.debug:
            st.write("pdfs path:", pdf_path)

        pdf_manager, retrievers = vector_store_builder(pdf_path, config, uploaded)
        st.session_state.pdf_manager = pdf_manager
        st.session_state.retrievers = retrievers

        # Create QA chains with current LLM manager
        st.session_state.qa_chains = QAchains(retrievers, config, llm_manager)
        st.success("PDFs and vector store processed successfully!")

    # Always ensure QA chains exist if we have retrievers and LLM manager
    if (
        st.session_state.get("retrievers") is not None
        and st.session_state.get("llm_manager") is not None
        and st.session_state.get("qa_chains") is None
    ):

        st.session_state.qa_chains = QAchains(
            st.session_state.retrievers, config, st.session_state.llm_manager
        )
        st.info("QA system initialized with selected model!")

    # ==============================
    # Question Section (only if retriever is successfully created)
    # ==============================
    if st.session_state.get("retrievers") is not None:
        question, answer = question_input_output_ui(st.session_state.qa_chains)

        if answer is not None:
            st.session_state.answer = answer
            # Store question, answer, and model info
            model_info = f"{selected_model['name']} ({selected_model['provider']})"
            st.session_state.qa_history.append((question, answer, model_info))
            logger.info(
                "Question answered: %s, answer: %s, model: %s",
                question,
                answer,
                model_info,
            )

    # ==============================
    # Display answer & history
    # ==============================
    display_results_ui(
        answer=st.session_state.answer,
        qa_history=st.session_state.qa_history,
    )
    logger.info("Displayed results and history.")


if __name__ == "__main__":
    main()
