# ---
# lambda-test: false  # auxiliary-file
# ---
# ## Demo Streamlit application.
#
# This application is the example from https://docs.streamlit.io/library/get-started/create-an-app.
#
# Streamlit is designed to run its apps as Python scripts, not functions, so we separate the Streamlit
# code into this module, away from the Modal application code.


def main():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import os
    from pathlib import Path
    from omegaconf import OmegaConf
    import sys
    # from PIL import Image
    # sys.path.append('/root/frontend')  # Add this line
    from backend.my_lib.pdf_manager import PDFManager
    from backend.my_lib.retrievers import Retrievers
    from backend.my_lib.qa_chains import QAchains
    from backend.settings import load_and_validate_env_secrets
    from backend.my_lib.LLMManager import LLMManager
    from frontend.helper_gui import (
        question_input_output_ui,
        display_results_ui,
        pdf_uploader_ui,
        select_model_ui,
        get_in_memory_mode,
        get_deployment_mode,
    )

    # ====================================
    # Initialize and clear problematic state on startup
    # ====================================
    if 'app_initialized' not in st.session_state:
        # Clear any media-related session state
        for key in list(st.session_state.keys()):
            if any(word in key.lower() for word in ['file', 'upload', 'media', 'image']):
                del st.session_state[key]
        
        # Clear all cached data to prevent 404 errors
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        st.session_state.app_initialized = True
        st.rerun()  # Single page reload for all cleanup    

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

    # logging from backend
    import logging

    logger = logging.getLogger(__name__)

    # ====================================
    # Initialize Streamlit session state variables
    # ====================================
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
        # logger.debug("Session state initialized.")
    
    # Initialize session state variables
    initialize_session_state()
    logger.debug("Session state initialized.")

    # Display the image at the top of the app
    image_path = "/root/frontend/static/image.jpeg"
    try:
        # os.chdir("/root")  # Set working directory        
        # if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
            # st.write(f"Image found at: {image_path}")
        # else:
            # st.write(f"Image not found at: {image_path}")
    except Exception as e:
        # logger.error(f"Error displaying image: {e}")
        st.write(f"Error displaying image: {e}")

    # Load configuration using OmegaConf
    config = OmegaConf.load("configs/config.yaml")

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
    deployment_mode = os.getenv("DEPLOYMENT_MODE", "local")
    deployment_emoji = "🏠" if deployment_mode == "local" else "☁️"
    st.sidebar.info(
        f"{deployment_emoji} **Deployment Mode:** {deployment_mode.title()}"
    )
    st.sidebar.info(
        # f"""📊 **Storage Mode:** {get_in_memory_mode()}
        # {get_in_memory_mode() == True}
        # {get_in_memory_mode() == "true"}
        # {bool(get_in_memory_mode())==True}
        # """
        f"📊 **Storage Mode:** {'In-Memory' if os.getenv('IN_MEMORY', 'false').lower() == 'true' else 'Persistent'}"
    )

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
        st.session_state.answer = None

        if st.session_state.verbose:
            st.info(f"Model changed to: {selected_model['name']}")


    # Initialize LLM Manager based on selected model
    if st.session_state.llm_manager is None or model_changed:    
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

        # CLEAR ANSWER WHEN PROCESSING NEW PDFs
        st.session_state.answer = None

        # Build vector store
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
    logger.debug("Displayed results and history.")    
    
    # DATE_COLUMN = "date/time"
    # DATA_URL = (
    #     "https://s3-us-west-2.amazonaws.com/"
    #     "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    # )

    # @st.cache_data
    # def load_data(nrows):
    #     data = pd.read_csv(DATA_URL, nrows=nrows)

    #     def lowercase(x):
    #         return str(x).lower()

    #     data.rename(lowercase, axis="columns", inplace=True)
    #     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    #     return data



    # data_load_state = st.text("Loading data...")
    # data = load_data(10000)
    # data_load_state.text("Done! (using st.cache_data)")

    # if st.checkbox("Show raw data"):
    #     st.subheader("Raw data")
    #     st.write(data)

    # st.subheader("Number of pickups by hour")
    # hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    # st.bar_chart(hist_values)

    # # Some number in the range 0-23
    # hour_to_filter = st.slider("hour", 0, 23, 17)
    # filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    # st.subheader("Map of all pickups at %s:00" % hour_to_filter)
    # st.map(filtered_data)


if __name__ == "__main__":
    main()