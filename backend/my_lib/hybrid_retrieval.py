import numpy as np
from langchain_community.retrievers import BM25Retriever
from backend.my_lib.pdf_manager import PDFManager
from backend.my_lib.retrievers import Retrievers

import logging

logger = logging.getLogger(__name__)


# ====================================
# Hybrid retrieval class for advanced document fusion
# ====================================
class Hybrid_Retrieval:
    """
    A class that implements hybrid retrieval combining BM25 keyword search and semantic search.

    This class provides advanced document retrieval capabilities by fusing results from
    multiple search strategies using Reciprocal Rank Fusion (RRF). It supports both
    hybrid and semantic-only retrieval modes for flexible document search.

    Attributes:
        pdf_manager (PDFManager): PDF document manager instance
        chunks (list): Large document chunks for retrieval
        vectorstore: Vector store for semantic search
        CE_model_keywords: Cross-encoder model for keyword search scoring
        CE_model_semantic: Cross-encoder model for semantic search scoring
        verbose (bool): Enable verbose logging
        modelID (str): OpenAI model identifier
        top_score_docs (list): Final ranked documents after fusion
    """

    # ====================================
    # Initialize hybrid retrieval system
    # ====================================
    def __init__(self, pdf_manager: PDFManager, retrievers: Retrievers, config):
        self.pdf_manager = pdf_manager
        self.chunks = pdf_manager.large_chunks
        self.vectorstore = pdf_manager.vectorstore
        self.CE_model_keywords = retrievers.CE_model_keywords
        self.CE_model_semantic = retrievers.CE_model_semantic
        self.verbose = config.settings.verbose
        self.modelID = config.llm.openai_modelID
        self.top_score_docs = None

    # ====================================
    # Perform hybrid retrieval with BM25 and semantic search fusion
    # ====================================
    def hybrid_retriever(
        self, question, top_k_BM25, top_k_semantic, top_k_final, rrf_k=60, hybrid=True
    ):
        """
        Perform hybrid document retrieval using BM25 and semantic search with RRF fusion.

        This method combines keyword-based BM25 retrieval with semantic vector search,
        then applies Reciprocal Rank Fusion (RRF) to merge and rank the results.

        Args:
            question (str): User query for document retrieval
            top_k_BM25 (int): Number of documents to retrieve via BM25
            top_k_semantic (int): Number of documents to retrieve via semantic search
            top_k_final (int): Final number of documents to return
            rrf_k (int, optional): RRF parameter for rank fusion. Defaults to 60.
            hybrid (bool, optional): Use hybrid mode (True) or semantic-only (False). Defaults to True.

        Returns:
            list[Document]: Top-ranked documents after fusion, limited to top_k_final
        """
        chunks = self.chunks

        if hybrid:
            logger.info("=== Hybrid Retrieval with BM25 and semantic search ===")
            retriever_kw = BM25Retriever.from_documents(documents=chunks, k=top_k_BM25)

            kw_chunks_retrieved = retriever_kw.invoke(question)
            if self.verbose:
                logger.info(
                    f"Number of retrieved documents: {len(kw_chunks_retrieved)}"
                )
                for chunk in kw_chunks_retrieved:
                    logger.info(
                        f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}'
                    )

            scores = self.CE_model_keywords.predict(
                [(question, result.page_content) for result in kw_chunks_retrieved]
            )

            # create a variable rank as the list of indices of largest scores items till the smallest
            rank_kw_chunks_retrieved = np.argsort(-scores)

            rank_kw_dict = {
                chunk.metadata["index"]: rank + 1
                for chunk, rank in zip(kw_chunks_retrieved, rank_kw_chunks_retrieved)
            }
            if self.verbose:
                logger.info("keyword retrieval scores:")
                logger.info(scores)
                logger.info("keyword retrieval ranks:")
                logger.info(rank_kw_dict)
        else:
            logger.info("=== Semantic search retrieval only === ")

        # Retrieval with semantic search
        retriever_large = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k_semantic}
        )
        large_chunks_retrieved = retriever_large.invoke(question)

        passages = [doc.page_content for doc in large_chunks_retrieved]
        ranks = self.CE_model_semantic.rank(question, passages)
        rank_large_chunks_retrieved = [rank["corpus_id"] for rank in ranks]

        if self.verbose:
            logger.info(f"Number of retrieved documents: {len(large_chunks_retrieved)}")
            for chunk in large_chunks_retrieved:
                logger.info(
                    f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}'
                )

        rank_semantic_dict = {
            chunk.metadata["index"]: rank + 1
            for chunk, rank in zip(large_chunks_retrieved, rank_large_chunks_retrieved)
        }

        if hybrid:
            # Calculate RRF score for the chunks
            rrf_scores = [0 for _ in range(len(chunks))]
            for index in range(len(chunks)):
                kw_rank = (
                    rank_kw_dict[index]
                    if index in rank_kw_dict.keys()
                    else float("inf")
                )
                semantic_rank = (
                    rank_semantic_dict[index]
                    if index in rank_semantic_dict.keys()
                    else float("inf")
                )
                rrf_scores[index] = (1 / (rrf_k + kw_rank)) + (
                    1 / (rrf_k + semantic_rank)
                )
            rrf_ranks = np.argsort(-np.array(rrf_scores))

            if self.verbose:
                logger.info("\nRRF scores:")
                logger.info(rrf_scores)
                logger.info("\nRRF ranks:")
                logger.info(rrf_ranks[:top_k_final])

            top_score_docs = list()
            for i in range(top_k_final):
                top_score_docs.append(chunks[rrf_ranks[i]])
        else:
            top_score_docs = list()
            for i in range(top_k_final):
                top_score_docs.append(
                    large_chunks_retrieved[rank_large_chunks_retrieved[i]]
                )

        if self.verbose:
            logger.info(f"Number of retrieved documents: {len(top_score_docs)}")
            for chunk in top_score_docs:
                logger.info(
                    f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}'
                )

        self.top_score_docs = top_score_docs
        return top_score_docs
