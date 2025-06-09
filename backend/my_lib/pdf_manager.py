import os
import streamlit as st
from backend.settings import is_streamlit_running
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from omegaconf import OmegaConf
import chromadb

import logging

logger = logging.getLogger(__name__)


# ------------------------------
# Define the PDFManager class
# ------------------------------
class PDFManager:
    """
    A class that manages the loading, processing, and storage of PDF documents for question answering.

    This class handles:
    1. Loading PDF files from a specified directory
    2. Splitting documents into small and large chunks for different retrieval strategies
    3. Creating and managing a vector store for semantic search
    4. Persisting document embeddings for efficient retrieval

    The class implements a two-stage chunking strategy to support both keyword-based
    and semantic retrieval methods, optimizing for both precision and recall in
    document search operations.
    """

    def __init__(self, pdf_path: str, config: OmegaConf):
        """
        Initializes the PDFManager with the necessary configurations.

        Args:
            pdf_path (str): Path to the directory containing PDF files.
            config (OmegaConf): Configuration object containing model and vector store settings.

        Attributes:
            pdf_path (str): Stores the path to the PDF directory.
            embed_model_id (str): ID of the embedding model to be used.
            persist_directory (str): Directory for persisting the vector store.
            collection_name (str): Name of the collection in the vector store.
            documents (list): List to store loaded documents.
            vectorstore (Optional): Vector store object, initialized as None.
            small_chunks (Optional): Placeholder for small document chunks, initialized as None.
            large_chunks (Optional): Placeholder for large document chunks, initialized as None.
        """
        self.pdf_path = pdf_path
        # self.config = config
        self.embed_model_id = config.llm.embed_model_id
        self.persist_directory = config.Vectorstore.persist_directory
        self.collection_name = config.Vectorstore.collection_name
        self.small_chunk_size = config.splitter.small_chunk_size
        self.large_chunk_size = config.splitter.large_chunk_size
        self.paragraph_separator = config.splitter.paragraph_separator
        self.documents = []
        self.vectorstore = None
        self.small_chunks = None
        self.large_chunks = None

    def load_pdfs(self) -> None:
        """
        Loads all PDF files from the specified directory using LangChain's PyPDFLoader.

        This method:
        1. Scans the specified directory for PDF files
        2. Loads each PDF using PyPDFLoader
        3. Adds metadata (filename and page number) to each document
        4. Combines all documents into a single list

        Attributes:
            documents (list): List of loaded documents, updated in-place.

        Raises:
            Exception: If PDF loading fails, displays error message via Streamlit or console.
        """
        try:
            filenames = [
                file
                for file in os.listdir(self.pdf_path)
                if file.lower().endswith(".pdf")
            ]
            if not filenames:
                if is_streamlit_running():
                    st.warning("No PDF files found in the specified directory.")
                else:
                    logger.warning("No PDF files found in the specified directory.")
                return

            docs = []
            for idx, file in enumerate(filenames):
                loader = PyPDFLoader(f"{self.pdf_path}/{file}")
                document = loader.load()
                for page_num, document_fragment in enumerate(document, start=1):
                    document_fragment.metadata = {"name": file, "page": page_num}

                # print(f'{len(document)} {document}\n')
                docs.extend(document)
            self.documents = docs
            if is_streamlit_running():
                st.success(
                    f"Total document pages loaded: {len(self.documents)} from {self.pdf_path}"
                )
            else:
                logger.info(
                    f"Total document pages loaded: {len(self.documents)} from {self.pdf_path}"
                )
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to load PDF files: {e}")
            else:
                logger.error(f"Failed to load PDF files: {e}")
            return

    def chunk_documents(self) -> None:
        """
        Splits loaded documents into small and large chunks using LangChain's RecursiveCharacterTextSplitter.

        Splits are performed with two different configurations: smaller chunks with no overlap and larger chunks with some overlap.

        Attributes:
            small_chunks (list): Stores the smaller document chunks.
            large_chunks (list): Stores the larger document chunks.

        Raises:
            Exception: If the document splitting process fails, an error message is displayed.
        """
        if not self.documents:
            if is_streamlit_running():
                st.error("No documents to split. Please load PDFs first.")
            else:
                logger.warning("No documents to split. Please load PDFs first.")
            return

        try:
            child_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.small_chunk_size,
                chunk_overlap=0,
                separators=["\n\n", "\n"],
            )
            self.small_chunks = child_text_splitter.split_documents(self.documents)
            # print(len(self.small_chunks), len(self.small_chunks[0].page_content))

            # Use paragraph separator if you know it from your documents formats
            large_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.large_chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n"],
            )
            large_chunks = large_text_splitter.split_documents(self.documents)
            for idx, chunk in enumerate(large_chunks):
                chunk.metadata["index"] = idx
            self.large_chunks = large_chunks
            # print(len(self.large_chunks), len(self.large_chunks[0].page_content))
            if is_streamlit_running():
                st.success(
                    f"Documents split into {len(self.small_chunks)} small and {len(self.large_chunks)} large chunks."
                )
            else:
                logger.info(
                    f"Documents split into {len(self.small_chunks)} small and {len(self.large_chunks)} large chunks."
                )
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to split documents: {e}")
            else:
                logger.error(f"Failed to split documents: {e}")

    def create_vectorstore(self) -> None:
        """
        Creates a vector store from the loaded document chunks using Chroma and HuggingFace embeddings.

        This function initializes an embedding model and a persistent Chroma client. It attempts to delete any existing
        collection with the specified collection name before creating a new vector store. The vector store is created
        using the large document chunks and is stored persistently.

        Attributes:
            vectorstore (Chroma): The created vector store containing the document embeddings.

        Raises:
            Exception: If there is an error during the creation of the vector store, an error message is displayed.
        """
        if not self.documents:
            if is_streamlit_running():
                st.error("No documents to index. Please load PDFs first.")
            else:
                logger.warning("No documents to index. Please load PDFs first.")
            return

        try:
            embedding = HuggingFaceEmbeddings(model_name=self.embed_model_id)
            # print(len(self.large_chunks))
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            try:
                chroma_client.delete_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} is deleted")
            except Exception:
                logger.warning(f"Collection {self.collection_name} does not exist")
            # print(len(chunks))
            self.vectorstore = Chroma.from_documents(
                documents=self.large_chunks,
                embedding=embedding,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

            collection = chroma_client.get_collection(name=self.collection_name)
            # st.success(f'Collection {collection_name} is created, number of itmes: {collection.count()}')
            if is_streamlit_running():
                st.success(
                    f"Vectorstore {self.collection_name} created successfully with {collection.count()} documents."
                )
            else:
                logger.info(
                    f"Vectorstore {self.collection_name} created successfully with {collection.count()} documents."
                )
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to create vectorstore: {e}")
            else:
                logger.error(f"Failed to create vectorstore: {e}")
