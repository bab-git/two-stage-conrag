# scripts/vstore_creator.py

import hydra
from omegaconf import DictConfig
from backend.settings import get_env_secrets
from backend.my_lib.pdf_manager import PDFManager


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("[INFO] Starting vectorstore creation test...")

    # Load API keys or secrets if needed
    secrets = get_env_secrets()
    print("[INFO] Loaded secrets:", list(secrets.keys()))

    # Set up PDF manager and run preprocessing pipeline
    pdf_manager = PDFManager(pdf_path=cfg.paths.pdf_path, config=cfg)
    pdf_manager.load_pdfs()
    pdf_manager.chunk_documents()
    pdf_manager.create_vectorstore()

    # print("[INFO] Vectorstore created and persisted at:", cfg.Vectorstore.persist_directory)


if __name__ == "__main__":
    main()
