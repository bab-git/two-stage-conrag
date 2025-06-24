# scripts/vstore_creator.py

"""
Vector store creation utility script for Two-Stage RAG system.

This script provides a standalone utility for creating and persisting
vector stores from PDF documents using Hydra configuration management.
"""

import hydra
from omegaconf import DictConfig
from backend.settings import load_and_validate_env_secrets
from backend.my_lib.pdf_manager import PDFManager


# ====================================
# Main vector store creation function
# ====================================
@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Create vector store from PDF documents using Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object with PDF paths and settings

    Note:
        This function expects the configuration to contain PDF path settings
        and proper environment variables for API keys.
    """
    print("[INFO] Starting vectorstore creation test...")

    # Load API keys or secrets if needed
    load_and_validate_env_secrets()
    print("[INFO] Environment secrets validated")

    # Set up PDF manager and run preprocessing pipeline
    pdf_manager = PDFManager(pdf_path=cfg.paths.pdf_path, config=cfg)
    pdf_manager.load_pdfs()
    pdf_manager.chunk_documents()
    pdf_manager.create_vectorstore()

    print("[INFO] Vectorstore created and persisted successfully")


if __name__ == "__main__":
    main()
