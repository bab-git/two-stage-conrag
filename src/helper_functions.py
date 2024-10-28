import os
from collections import defaultdict
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import numpy as np
from pprint import pprint
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever                
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import SelfQueryRetriever
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb


def is_streamlit_running():
    """
    Checks if the script is running within a Streamlit app.

    Returns:
        bool: True if running in Streamlit, False otherwise.
    """    
    try:        
        return get_script_run_ctx() is not None
    except:
        return False

sreamlit_running = is_streamlit_running()
if sreamlit_running == False:
    print('streamlit is not running')

# ------------------------------
# Define the PDFManager class
# ------------------------------
class PDFManager:
    """
    Manages PDF loading, chunking, and vector store creation.
    """
    def __init__(self, pdf_path: str, config):
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

    def load_pdfs(self):            
        """
        Loads all PDF files from the specified directory using LangChain's PyPDFLoader.        

        Attributes:
            documents (list): List of loaded documents, updated in-place.
        """
        # filenames = os.listdir(self.pdf_path)
        # print(filenames)
        # metadata = [dict(source=filename) for filename in filenames]

        try:
            filenames = [file for file in os.listdir(self.pdf_path) if file.lower().endswith('.pdf')]
            if not filenames:
                st.warning("No PDF files found in the specified directory.")
                return

            docs = []
            for idx, file in enumerate(filenames):
                loader = PyPDFLoader(f'{self.pdf_path}/{file}')
                document = loader.load()
                for page_num, document_fragment in enumerate(document, start=1):
                    document_fragment.metadata = {"name": file, "page": page_num}
                    
                # print(f'{len(document)} {document}\n')
                docs.extend(document)
            self.documents = docs            
            if sreamlit_running:
                st.success(f"Total document pages loaded: {len(self.documents)} from {self.pdf_path}")
        except Exception as e:
            st.error(f"Failed to load PDF files: {e}")
            return

    def chunk_documents(self):        
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
            st.error("No documents to split. Please load PDFs first.")
            return
        
        try:
            child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.small_chunk_size, chunk_overlap=0, separators=["\n\n", "\n"])
            self.small_chunks = child_text_splitter.split_documents(self.documents)
            # print(len(self.small_chunks), len(self.small_chunks[0].page_content))

            # Use paragraph separator if you know it from your documents formats
            large_text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.large_chunk_size, chunk_overlap=200, separators=["\n\n", "\n"])
            large_chunks = large_text_splitter.split_documents(self.documents)
            for idx, chunk in enumerate(large_chunks):
                chunk.metadata['index'] = idx
            self.large_chunks = large_chunks
            # print(len(self.large_chunks), len(self.large_chunks[0].page_content))
            if sreamlit_running:
                st.success(f"Documents split into {len(self.small_chunks)} small and {len(self.large_chunks)} large chunks.")
        except Exception as e:
            st.error(f"Failed to split documents: {e}")
    
    def create_vectorstore(self):
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
            st.error("No documents to index. Please load PDFs first.")
            return

        try:            
            embedding = HuggingFaceEmbeddings(model_name=self.embed_model_id)                                    
            # print(len(self.large_chunks))
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            try:
                chroma_client.delete_collection(self.collection_name)
                print(f'Collection {self.collection_name} is deleted')
            except Exception:
                print(f'Collection {self.collection_name} does not exist')
            # print(len(chunks))
            self.vectorstore = Chroma.from_documents(
                documents=self.large_chunks,
                embedding=embedding,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

            collection = chroma_client.get_collection(name=self.collection_name)
            # st.success(f'Collection {collection_name} is created, number of itmes: {collection.count()}')
            if sreamlit_running:
                st.success(f"Vectorstore {self.collection_name} created successfully with {collection.count()} documents.")
        except Exception as e:
            st.error(f"Failed to create vectorstore: {e}")
  
# ------------------------------
# Define the QA Retriever Function
# ------------------------------
class Retrievers:
    """
    Sets up various retrievers for keyword and semantic search.
    """
    def __init__(self, pdf_manager: PDFManager, config):        
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

    def setup_retrievers(self):
        """
        Sets up the retrievers.

        Sets up the BM25 retriever and the large retriever based on the vectorstore and small chunks of documents.
        """
        if not self.small_chunks:
            st.error("No small_chunks to index. Please load PDFs first.")
            return
        try:
            self.retriever_small = BM25Retriever.from_documents(documents=self.small_chunks,k=self.top_k_BM25)
            self.CE_model_keywords = CrossEncoder(self.keyword_CE_model)
            self.CE_model_semantic = CrossEncoder(self.semantic_CE_model)

            if sreamlit_running:
                st.success("Retrievers created successfully.")
        except Exception as e:
            st.error(f"Failed to create retrievers: {e}")
    
    def retrieve_small_chunks(self, shortened_question):          
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
                chunk.metadata['score'] = scores[i]
            
            # for debugging:            
            if self.verbose:
                print('\n',len(small_chunks_retrieved), 'small chunks retrieved')
                print('\n ==== Samples of retrieved small chunks ==== \n')
                for chunk in small_chunks_retrieved[:3]:
                    print(chunk.metadata['name'], f"page:{chunk.metadata['page']}", 
                            f"score:{chunk.metadata['score']}", chunk.page_content[:20])

            return small_chunks_retrieved

        except Exception as e:
            st.error(f"Failed to retrieve small chunks: {e}")
            return None

    def calculate_drs(self, small_chunks_retrieved): 
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
            st.error(f"Failed to calculate DRS for PDF documents: {e}")
            return None    
    
    def score_aggregate(self, retrieved_chunks, normalized_drs):
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
            st.error(f"Failed to aggregate similarity scores for retrieved chunks: {e}")
            return None
                        
    def retrieve_large_chunks(self, question, files):
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
            st.error(f"Failed to retrieve large chunks: {e}")
            return None

class QAchains:
    """
    Handles the Question-Answering pipeline, including question shortening, retrieval, ranking, and answer generation.
    """
    def __init__(self, retrievers: Retrievers, config):        
        """
        Initializes the QAchain object.

        Args:
            retrievers (Retrievers): The object containing the retrievers.
            config (Config): The configuration object containing the necessary settings.
        """
                
        self.top_k_final = config.Retrieval.top_k_final
        self.verbose = config.settings.verbose
        self.retrievers = retrievers
        modelID = config.llm.openai_modelID
        self.llm = ChatOpenAI(temperature = 0.0, model=modelID)
        self.question = None
        self.shortened_question = None
        self.retrieved_docs = None
        self.top_score_docs = None

    def shorten_question(self, question: str):
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

        Reformulated phrases: """
        
        try:                        
            custom_short_prompt = PromptTemplate.from_template(shortening_prompt)
                        
            shortening_chain = (
                {"original_question": RunnablePassthrough()}
                | custom_short_prompt
                | self.llm
                | StrOutputParser()
            )

            shortened_question = shortening_chain.invoke(question)
            print(shortened_question)
            if sreamlit_running:
                st.success(f"The shortened question:\n {shortened_question}")
            
            self.question = question
            self.shortened_question = shortened_question

        except Exception as e:
            st.error(f"Failed to generate shortened question: {e}")
    
    def retrieve_context(self):
        try:
            question = self.question
            shortened_question = self.shortened_question
            top_k_final = self.top_k_final

            # retrieve relevant small chunks
            small_chunks_retrieved = self.retrievers.retrieve_small_chunks(shortened_question)
            if sreamlit_running:
                st.success("The small chunks were retrieved")

            # Calculate DRS for all documents
            documents_selected, DRS_selected_normalized = self.retrievers.calculate_drs(small_chunks_retrieved)
            if sreamlit_running:
                st.success("The DRS was calculated for relevant PDF documents")

            # retrieve relevant large chunks
            large_chunks_retrieved = self.retrievers.retrieve_large_chunks(question, documents_selected)
            if sreamlit_running:
                st.success("The large chunks were retrieved")
            
            # Calculate aggregated scores for small and large chunks
            small_chunks_agg_score = self.retrievers.score_aggregate(small_chunks_retrieved, DRS_selected_normalized)
            if self.verbose:
                print("\n === The aggregated scores were calculated for relevant small chunks ==")
                for doc in small_chunks_agg_score:
                    print(doc.metadata['aggregated_score'], doc.metadata['name'], doc.metadata['page'], doc.page_content[:20])
            
            large_chunks_agg_score = self.retrievers.score_aggregate(large_chunks_retrieved, DRS_selected_normalized)
            
            if self.verbose:
                print("\n === The aggregated scores were calculated for relevant large chunks ==")
                for doc in large_chunks_agg_score:
                    print(doc.metadata['aggregated_score'], doc.metadata['name'], doc.metadata['page'], doc.page_content[:20])

            if sreamlit_running:
                st.success("The aggregated scores were calculated for all retrieved chunks")

            # concatenate best small and large chunks
            top_score_docs = large_chunks_agg_score[:top_k_final] + small_chunks_agg_score[:top_k_final][::-1]
            
            if self.verbose:
                print("\n === The top score chunks were concatenated ==")
                for doc in top_score_docs:
                    print(doc.metadata['score'], doc.metadata['name'], doc.metadata['page'], doc.page_content[:20])            

            if sreamlit_running:
                st.success("The top score chunks were concatenated")

            self.top_score_docs = top_score_docs

        except Exception as e:
            st.error(f"Failed to retrieve context for the input quersion: {e}")            

    def generate_answer(self):
        """
        Generate an answer to the question based on the top-k ranked chunks of documents.

        Uses the LangChain's RetrievalQA to generate an answer based on the top-k ranked chunks of documents.
        The answer is generated using a custom prompt template that provides context from the top-k ranked documents.
        The answer is then parsed and returned as a string.

        :return: str
        """

        template = """ You are an expert financial analyst with extensive experience in interpreting reports, analyzing financial data, and generating insights from dense textual information. 
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
            custom_rag_prompt = PromptTemplate.from_template(template)
            chain = custom_rag_prompt | self.llm | StrOutputParser()
            
            context = "\ndocument_separator: <<<<>>>>>\n".join(doc.page_content for doc in self.top_score_docs)
            response = chain.invoke({"context": context, "question": self.question})
            return response.strip()
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")

class Hybrid_Retrieval:
    def __init__(self, pdf_manager: PDFManager, retrievers: Retrievers, config):
        self.pdf_manager = pdf_manager
        self.chunks = pdf_manager.large_chunks
        self.vectorstore = pdf_manager.vectorstore
        self.CE_model_keywords = retrievers.CE_model_keywords
        self.CE_model_semantic = retrievers.CE_model_semantic        
        self.verbose = config.settings.verbose
        self.modelID = config.llm.openai_modelID 
        self.top_score_docs = None       
    
    def hybrid_retriever(self, question, top_k_BM25, top_k_semantic, top_k_final, rrf_k = 60, hybrid = True):
        chunks = self.chunks
                
        if hybrid:
            print('=== Hybrid Retrieval with BM25 and semantic search ===')
            retriever_kw = BM25Retriever.from_documents(documents = chunks, k=top_k_BM25)        

            kw_chunks_retrieved = retriever_kw.invoke(question) 
            if self.verbose:
                print(f'Number of retrieved documents: {len(kw_chunks_retrieved)}')
                for chunk in kw_chunks_retrieved:
                    print(f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}')

            scores = self.CE_model_keywords.predict(
                [(question, result.page_content) for result in kw_chunks_retrieved]
            )        

            # create a variable rank as the list of indices of largest scores items till the smallest
            rank_kw_chunks_retrieved = np.argsort(-scores) 
            
            rank_kw_dict = {chunk.metadata['index'] : rank + 1 for chunk, rank in zip(kw_chunks_retrieved, rank_kw_chunks_retrieved)}    
            if self.verbose:
                print('keyword retrieval scores:')
                print(scores)      
                print('keyword retrieval ranks:')
                print(rank_kw_dict)
        else:
            print('=== Semantic search retrieval only === ')

        # Retrieval with semantic search
        retriever_large = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k_semantic})
        large_chunks_retrieved = retriever_large.invoke(question)        
                
        passages = [doc.page_content for doc in large_chunks_retrieved]
        ranks = self.CE_model_semantic.rank(question, passages)
        rank_large_chunks_retrieved = [rank['corpus_id'] for rank in ranks]
        
        if self.verbose:
            print(f'Number of retrieved documents: {len(large_chunks_retrieved)}')
            for chunk in large_chunks_retrieved:
                print(f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}')
        
        rank_semantic_dict = {chunk.metadata['index']: rank + 1 for chunk, rank in zip(large_chunks_retrieved, rank_large_chunks_retrieved)}        

        if hybrid:
            # Calculate RRF score for the chunks
            rrf_scores = [0 for _ in range(len(chunks))]
            for index in range(len(chunks)):
                kw_rank = rank_kw_dict[index] if index in rank_kw_dict.keys() else float('inf')
                semantic_rank = rank_semantic_dict[index] if index in rank_semantic_dict.keys() else float('inf')                        
                rrf_scores[index] = (1 / (rrf_k + kw_rank)) + (1 / (rrf_k + semantic_rank))            
            rrf_ranks = np.argsort(-np.array(rrf_scores))

            if self.verbose:
                print('\nRRF scores:')
                print(rrf_scores)
                print('\nRRF ranks:')
                print(rrf_ranks[:top_k_final])

            top_score_docs = list()
            for i in range(top_k_final):
                top_score_docs.append(chunks[rrf_ranks[i]])
        else:
            top_score_docs = list()
            for i in range(top_k_final):
                top_score_docs.append(large_chunks_retrieved[rank_large_chunks_retrieved[i]])
        
        if self.verbose:
            print(f'Number of retrieved documents: {len(top_score_docs)}')
            for chunk in top_score_docs:
                print(f'name: {chunk.metadata["name"]}, page: {chunk.metadata["page"]}, page_content: {chunk.page_content[:10]}')

        self.top_score_docs = top_score_docs
        return top_score_docs