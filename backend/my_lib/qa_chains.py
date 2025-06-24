import streamlit as st
from backend.my_lib.retrievers import Retrievers
from backend.settings import is_streamlit_running
from omegaconf import OmegaConf
from backend.my_lib.LLMManager import LLMManager

import logging

logger = logging.getLogger(__name__)


# ====================================
# The QAchains class
# ====================================
class QAchains:
    """
    A class that orchestrates the Question-Answering pipeline for document-based queries.

    This class implements a multi-stage QA process that:
    1. Shortens user questions to extract key terms and entities
    2. Retrieves relevant document chunks using hybrid retrieval
    3. Ranks and filters retrieved documents based on relevance scores
    4. Generates comprehensive answers using retrieved context

    The pipeline combines keyword-based and semantic retrieval methods to ensure
    both precision and recall in document retrieval, while maintaining efficiency
    through question shortening and document ranking.
    """

    # ====================================
    # Initialize QA chains with retrievers and configuration
    # ====================================
    def __init__(
        self, retrievers: Retrievers, config: OmegaConf, llm_manager: LLMManager = None
    ):
        """
        Initializes the QAchain object.

        Args:
            retrievers (Retrievers): The object containing the retrievers.
            config (Config): The configuration object containing the necessary settings.
        """

        self.top_k_final = config.Retrieval.top_k_final
        self.verbose = config.settings.verbose
        self.retrievers = retrievers
        if llm_manager is None:
            # a default model config for fallback
            default_model_config = {
                "name": "GPT-4o-mini",
                "model_id": "gpt-4o-mini",
                "provider": "openai",
            }
            self.llm_manager = LLMManager(default_model_config)
        else:
            self.llm_manager = llm_manager
        self.question = None
        self.shortened_question = None
        self.retrieved_docs = None
        self.top_score_docs = None
        self.response = None
        self.verbose = config.settings.verbose
        self.selected_documents = None
        self.drs_scores = None

    # ====================================
    # Shorten question to essential keywords using LLM
    # ====================================
    def shorten_question(self, question: str) -> None:
        """
        Shortens the question to a short phrase with essential keywords.

        Uses a ChatLLM to generate a shortened version of the question. The prompt is a description of the task of
        shortening the question with essential keywords. The shortened question is then used to retrieve relevant
        documents.

        Args:
            question (str): The original question to be shortened.

        Raises:
            Exception: If there is an error during the generation of the shortened question, an error message is displayed.
        """

        shortening_prompt = """
        You are an expert financial advisor tasked with shortening the original question. 
        Your role is to reformulate the original question to short phrases with essential keywords.
        Mostly focus on company names, consultant or advisor names.
        The answer does not need to be complete sentense.
        Do not convert words to abbreviations.

        Original Question: "{original_question}"

        Reformulated phrases: 
        """

        try:
            invoke_kwargs = {"original_question": question}
            shortened_question = self.llm_manager.invoke(
                shortening_prompt, invoke_kwargs, max_tokens=128, verbose=self.verbose
            )
            if is_streamlit_running():
                st.success(f"The shortened question:\n {shortened_question}")
            else:
                logger.info(f"The shortened question:\n {shortened_question}")

            self.question = question
            self.shortened_question = shortened_question

        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to generate shortened question: {e}")
            else:
                logger.error(f"Failed to generate shortened question: {e}")

    # ====================================
    # Retrieve and process relevant context using two-stage approach
    # ====================================
    def retrieve_context(self) -> None:
        """
        Retrieves and processes relevant context from documents using a two-stage retrieval approach.

        This method implements a hybrid retrieval strategy that combines:
        1. Initial retrieval of small chunks using the shortened question
        2. Document-level scoring (DRS) calculation
        3. Retrieval of large chunks based on the original question and filtered documents
        4. Score aggregation for both small and large chunks
        5. Final concatenation of the highest-scoring chunks

        The method updates the instance variables with the retrieved context and scores.

        Raises:
            Exception: If any step in the retrieval process fails, an error message is displayed.
        """
        try:
            question = self.question
            shortened_question = self.shortened_question
            top_k_final = self.top_k_final

            # retrieve relevant small chunks
            small_chunks_retrieved = self.retrievers.retrieve_small_chunks(
                shortened_question
            )
            if is_streamlit_running():
                st.success(f"{len(small_chunks_retrieved)} small chunks were retrieved")
            else:
                logger.info(
                    f"{len(small_chunks_retrieved)} small chunks were retrieved"
                )

            # Calculate DRS for all documents
            documents_selected, DRS_selected_normalized = self.retrievers.calculate_drs(
                small_chunks_retrieved
            )

            # Store the selected documents and their scores
            self.selected_documents = documents_selected
            self.drs_scores = DRS_selected_normalized

            if is_streamlit_running():
                st.success("The DRS was calculated for relevant PDF documents")

                # Display selected documents here - after DRS calculation
                if self.selected_documents:
                    with st.expander(
                        f"📄 Selected Documents ({len(self.selected_documents)})",
                        expanded=False,
                    ):
                        for i, doc_name in enumerate(self.selected_documents, 1):
                            score = self.drs_scores.get(doc_name, 0.0)
                            st.write(f"{i}. {doc_name} - DRS: {score:.3f}")
            else:
                logger.info("The DRS was calculated for relevant PDF documents")

            # retrieve relevant large chunks
            large_chunks_retrieved = self.retrievers.retrieve_large_chunks(
                question, documents_selected
            )
            if is_streamlit_running():
                st.success(f"{len(large_chunks_retrieved)} large chunks were retrieved")
            else:
                logger.info(
                    f"{len(large_chunks_retrieved)} large chunks were retrieved"
                )

            # Calculate aggregated scores for small and large chunks
            small_chunks_agg_score = self.retrievers.score_aggregate(
                small_chunks_retrieved, DRS_selected_normalized
            )
            if self.verbose:
                logger.info(
                    "\n === The aggregated scores were calculated for relevant small chunks =="
                )
                for doc in small_chunks_agg_score:
                    logger.info(
                        "Score: %s, Name: %s, Page: %s, Content: %s",
                        doc.metadata["aggregated_score"],
                        doc.metadata["name"],
                        doc.metadata["page"],
                        doc.page_content[:20],
                    )

            large_chunks_agg_score = self.retrievers.score_aggregate(
                large_chunks_retrieved, DRS_selected_normalized
            )

            if self.verbose:
                logger.info(
                    "\n === The aggregated scores were calculated for relevant large chunks =="
                )
                for doc in large_chunks_agg_score:
                    logger.info(
                        "Score: %s, Name: %s, Page: %s, Content: %s",
                        doc.metadata["aggregated_score"],
                        doc.metadata["name"],
                        doc.metadata["page"],
                        doc.page_content[:20],
                    )

            if is_streamlit_running():
                st.success(
                    "The aggregated scores were calculated for all retrieved chunks"
                )

                # Display top 3 chunks here - after aggregated scores calculation
                all_chunks = large_chunks_agg_score + small_chunks_agg_score
                top_3_chunks = sorted(
                    all_chunks,
                    key=lambda x: x.metadata["aggregated_score"],
                    reverse=True,
                )[:3]

                if top_3_chunks:
                    with st.expander(
                        "🏆 Top 3 Chunks (Highest Scores)", expanded=False
                    ):
                        for i, chunk in enumerate(top_3_chunks, 1):
                            score = chunk.metadata.get("aggregated_score", 0.0)
                            doc_name = chunk.metadata.get("name", "Unknown")
                            page = chunk.metadata.get("page", "N/A")
                            content_preview = (
                                chunk.page_content[:300] + "..."
                                if len(chunk.page_content) > 300
                                else chunk.page_content
                            )

                            st.write(f"**#{i} - Score: {score:.3f}**")
                            st.write(f"📄 Document: {doc_name} (Page: {page})")
                            st.write(f"📝 Content: {content_preview}")
                            if i < len(
                                top_3_chunks
                            ):  # Don't add separator after last item
                                st.write("---")
            else:
                logger.info(
                    "The aggregated scores were calculated for all retrieved chunks"
                )

            # concatenate best small and large chunks
            top_score_docs = (
                large_chunks_agg_score[:top_k_final]
                + small_chunks_agg_score[:top_k_final][::-1]
            )

            if self.verbose:
                logger.info("\n === The top score chunks were concatenated ==")
                for doc in top_score_docs:
                    logger.info(
                        "Score: %s, Name: %s, Page: %s, Content: %s",
                        doc.metadata["aggregated_score"],
                        doc.metadata["name"],
                        doc.metadata["page"],
                        doc.page_content[:20],
                    )

            if is_streamlit_running():
                st.success(f"{len(top_score_docs)} top score chunks were concatenated")
            else:
                logger.info(f"{len(top_score_docs)} top score chunks were concatenated")

            self.top_score_docs = top_score_docs

        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to retrieve context for the input quersion: {e}")
            else:
                logger.error(f"Failed to retrieve context for the input quersion: {e}")

    # ====================================
    # Generate final answer from retrieved context using LLM
    # ====================================
    def generate_answer(self) -> str:
        """
        Generate an answer to the question based on the top-k ranked chunks of documents.

        Uses the LangChain's RetrievalQA to generate an answer based on the top-k ranked chunks of documents.
        The answer is generated using a custom prompt template that provides context from the top-k ranked documents.
        The answer is then parsed and returned as a string.

        Returns:
            str: Generated answer based on retrieved context, or None if generation fails
        """
        system_prompt = """ You are an expert financial analyst with extensive experience in interpreting reports, analyzing financial data, and generating insights from dense textual information. 
        Your task is to answer questions using only the provided document chunks as context. 
        Your answers should focus solely on the information within the document chunks and avoid speculation or any information not directly supported by the text.
        The document context provided includes various financial reports, business analyses, and forecasting values. 
        Your role is to deliver concise, well-supported responses that draw from this context, aligning with the standards and depth of a financial consultant.
        Always mention any value or number you find in the context that is relevant to the question.
        Also mention any non-numeric information that can clarify the financial context related to the question.
        If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}

        Expert financial answer:
        """

        try:
            # custom_rag_prompt = PromptTemplate.from_template(system_prompt)
            # chain = custom_rag_prompt | self.llm | StrOutputParser()

            context = "\ndocument_separator: <<<<>>>>>\n".join(
                doc.page_content for doc in self.top_score_docs
            )
            # response = chain.invoke({"context": context, "question": self.question})
            invoke_kwargs = {"context": context, "question": self.question}
            response = self.llm_manager.invoke(system_prompt, invoke_kwargs)
            self.response = response
            return self.response
        except Exception as e:
            if is_streamlit_running():
                st.error(f"Failed to generate answer: {e}")
            else:
                logger.error(f"Failed to generate answer: {e}")
