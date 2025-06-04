import streamlit as st
from backend.my_lib.pdf_manager import PDFManager
from collections import defaultdict
import numpy as np
from langchain_community.retrievers import BM25Retriever                
from sentence_transformers.cross_encoder import CrossEncoder
from backend.settings import is_streamlit_running
from omegaconf import OmegaConf
from langchain_core.documents import Document

# streamlit_running = is_streamlit_running()

# ------------------------------
# Define the QA Retriever Function
# ------------------------------
class Retrievers:
    """
    A class that manages and coordinates different types of retrievers for document search.

    This class implements a hybrid retrieval system that combines:
    - BM25 keyword-based retrieval for initial document filtering
    - Semantic search using cross-encoders for relevance scoring
    - Document-level scoring and ranking

    The class handles the setup and coordination of these retrievers to provide
    efficient and accurate document retrieval for question answering tasks.
    """
    def __init__(self, pdf_manager: PDFManager, config: OmegaConf):        
        """
        Initialize the retriever with the vectorstore and small chunks of documents

        Args:
            pdf_manager (PDFManager): The PDFManager instance
            config (Config): The configuration object
        """

        self.vectorstore = pdf_manager.vectorstore
        self.small_chunks = pdf_manager.small_chunks
        # self.modelID = config.llm.openai_modelID
        self.top_k_BM25 = config.Retrieval.top_k_BM25   
        self.top_k_semantic = config.Retrieval.top_k_semantic 
        self.top_k_documents = config.Retrieval.top_k_documents
        self.semantic_CE_model = config.Retrieval.semantic_CE_model        
        self.keyword_CE_model = config.Retrieval.keyword_CE_model        
        self.verbose = config.settings.verbose
        self.retriever_small = None

    def setup_retrievers(self) -> None:
        """
        Sets up the retrievers.

        Sets up the BM25 retriever and the large retriever based on the vectorstore and small chunks of documents.
        """
        if not self.small_chunks:
            if is_streamlit_running():
                st.error("No small_chunks to index. Please load PDFs first.")
            else:
                print("No small_chunks to index. Please load PDFs first.")
            return
        try:
            self.retriever_small = BM25Retriever.from_documents(documents=self.small_chunks,k=self.top_k_BM25)
            self.CE_model_keywords = CrossEncoder(self.keyword_CE_model)
            self.CE_model_semantic = CrossEncoder(self.semantic_CE_model)

            if is_streamlit_running():
                st.success("Retrievers created successfully.")
            else:
                print("Retrievers created successfully.")
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to create retrievers: {e}")
            else:
                print(f"Failed to create retrievers: {e}")
    
    def retrieve_small_chunks(self, shortened_question: str) -> list[Document]:          
        """
        Retrieves relevant small chunks based on the shortened question 
        and calculates similarity scores using the cross-encoder model.

        Parameters
        ----------
        shortened_question : str
            The shortened version of the user's question.

        Returns
        -------
        List of Document objects
            The retrieved small chunks ordered by their similarity score.
        """
        try:
            # Retrieve the small chunks
            small_chunks_retrieved = self.retriever_small.invoke(shortened_question)            
            
            # Calculate similarity score for each chunk
            scores = self.CE_model_keywords.predict(
                [(shortened_question, result.page_content) for result in small_chunks_retrieved]
            )

            for i, chunk in enumerate(small_chunks_retrieved):
                chunk.metadata['score'] = float(scores[i])
            
            # for debugging:            
            if self.verbose:
                print('\n',len(small_chunks_retrieved), 'small chunks retrieved')
                print('\n ==== Samples of retrieved small chunks ==== \n')
                for chunk in small_chunks_retrieved[:3]:
                    print(chunk.metadata['name'], f"page:{chunk.metadata['page']}", 
                            f"score:{chunk.metadata['score']}", chunk.page_content[:20])

            return small_chunks_retrieved

        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to retrieve small chunks: {e}")
            else:
                print(f"Failed to retrieve small chunks: {e}")
            return None

    def calculate_drs(self, small_chunks_retrieved: list[Document]) -> tuple[list[str], dict[str, float]]: 
        """
        Calculate the Document Retrieval Score (DRS) for the documents associated with the retrieved small chunks.

        The DRS is calculated as the sum of the similarity scores of the small chunks
        from each document, divided by the number of chunks from the same document.
        The DRS is then normalized by dividing by the maximum DRS score.

        Parameters
        ----------
        small_chunks_retrieved : List of Document objects
            The retrieved small chunks ordered by their similarity score.

        Returns
        -------
        List of str, dict
            A tuple containing the selected document names and their corresponding
            normalized DRS scores.
        """
        # print('small_chunks_retrieved', small_chunks_retrieved)

        pos_score = any(chunk.metadata['score'] > 0 for chunk in small_chunks_retrieved)
        if not pos_score:
            print('\n No small chunk retrieved with a positive score \n')
        
        DRS_documents = defaultdict(lambda: {'N': 0, 'sum_score': 0, 'score': 0})

        try:
            for chunk in small_chunks_retrieved:
                # print(score)
                name = chunk.metadata['name']
                # page = chunk.metadata['page']
                score = chunk.metadata['score']
                if score > 0:                    
                    DRS_documents[name]['N'] += 1
                    DRS_documents[name]['sum_score'] += max(0,score)
                    log_plus_1 = np.log(1 + DRS_documents[name]['N'])
                    DRS_documents[name]['score'] = DRS_documents[name]['sum_score'] *log_plus_1
                elif pos_score == False:
                    DRS_documents[name]['N'] += 1
                    score = 1/abs(score)
                    DRS_documents[name]['score'] = max(DRS_documents[name]['score'],score)

            # sort documents according to DRS
            DRS_documents_sorted = sorted(DRS_documents.items(), key=lambda item: item[1]['score'], reverse=True) 
            
            if self.verbose:
                print('\n ==== Sorted DRS documents ====')            
                for doc in DRS_documents_sorted:
                    print(doc[0], f'score:{doc[1]['score']}, N:{doc[1]["N"]}')
            
            # selecting documents
            documents_selected = [doc[0] for doc in DRS_documents_sorted[:self.top_k_documents]]

            if self.verbose:
                print('\n ==== Selected documents ====')
                for document in documents_selected:
                    print(document)

            # normalized DRS: divide DRS by max DRS
            DRS_selected = DRS_documents_sorted[:self.top_k_documents]
            max_drs_score = max(entry[1]['score'] for entry in DRS_selected)
            DRS_selected_normalized = {
                entry[0]: (entry[1]['score']) / (max_drs_score)
                for entry in DRS_selected
            }
            
            return documents_selected, DRS_selected_normalized
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to calculate DRS for PDF documents: {e}")
            else:
                print(f"Failed to calculate DRS for PDF documents: {e}")
            return None    
    
    def score_aggregate(self, retrieved_chunks: list[Document], normalized_drs: dict[str, float]) -> list[Document]:
        """
        Aggregate similarity scores for retrieved chunks by multiplying the chunk-level similarity score with the DRS score of the document.
        
        Args:
            retrieved_chunks (list[Document]): Retrieved chunks to aggregate scores for.
            normalized_drs (dict[str, float]): Normalized DRS scores for each document, where the key is the document name and the value is the normalized score.

        Returns:
            list[Document]: Retrieved chunks sorted in descending order of their aggregated scores.
        """
        try:
            for doc in retrieved_chunks:
                doc_name = doc.metadata['name']    
                chunk_score = doc.metadata.get('score', 0)
                drs_score = normalized_drs.get(doc_name, 0)  # Default to 0 if not found
                doc.metadata['aggregated_score'] = chunk_score * drs_score        

            #  Sort documents by aggregated score in descending order
            retrieved_docs_sorted = sorted(retrieved_chunks, key=lambda x: x.metadata['aggregated_score'], reverse=True)
            # retrieved_docs_selected = retrieved_docs_sorted[:self.top_k_final]

            return retrieved_docs_sorted
        
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to aggregate similarity scores for retrieved chunks: {e}")
            else:
                print(f"Failed to aggregate similarity scores for retrieved chunks: {e}")
            return None
                        
    def retrieve_large_chunks(self, question: str, files: list[str]) -> list[Document]:
        """
        Retrieve relevant large chunks from the vector store based on the given question and the filtered file names.

        Args:
            question (str): The user's question.
            files (list[str]): The filtered file names.

        Returns:
            list[Document]: The retrieved large chunks sorted in descending order of their similarity scores.
        """
        try:                                 
            # Create the filter for document names
            name_filter = {"name": {"$in": files}}
            large_chunks_retrieved = self.vectorstore.similarity_search(
                query=question,
                k = self.top_k_semantic,
                filter=name_filter
            )

            if self.verbose:
                print(f'\n ==== {len(large_chunks_retrieved)} Retrieved large chunks ==== \n')                
                for chunk in large_chunks_retrieved:
                    print(f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}')
            
            # Calculate similarity score for each chunk
            passages = [doc.page_content for doc in large_chunks_retrieved]
            ranks = self.CE_model_semantic.rank(question, passages)
            
            # print("question:", question)
            print('\n ==== Ranked retrieved large chunks ==== \n')
            for rank in ranks:
                large_chunks_retrieved[rank['corpus_id']].metadata['score'] = rank['score']
                if self.verbose:
                    print(f"{rank['score']:.2f}\t{large_chunks_retrieved[rank['corpus_id']].metadata}, {large_chunks_retrieved[rank['corpus_id']].page_content[:10]} ")
                
            return large_chunks_retrieved

        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to retrieve large chunks: {e}")
            else:
                print(f"Failed to retrieve large chunks: {e}")
            return None