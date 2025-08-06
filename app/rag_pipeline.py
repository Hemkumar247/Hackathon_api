# app/rag_pipeline.py (Updated for Multi-Agent System)

import os
import traceback
import requests
import tempfile
from typing import List

# Import our new agents and LangChain components
from .agents import get_router_agent, get_synthesizer_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load Foundational Embedding Model ---
try:
    print("--- RAG Pipeline: Loading embedding model... ---")
    # Using Google's embedding model, which uses the GOOGLE_API_KEY from the environment
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("--- RAG Pipeline: Google embedding model loaded successfully. ---")
except Exception as e:
    print(f"--- FATAL RAG ERROR: Could not load embedding model: {e} ---")
    traceback.print_exc()
    embedding_model = None

import logging

# --- Initialize the Agents ---
router_agent = get_router_agent()
synthesizer_agent = get_synthesizer_agent()


def process_rag_query(document_url: str, questions: List[str]) -> List[str]:
    if not embedding_model:
        logging.error("FATAL: Embedding model is not loaded.")
        raise RuntimeError("Embedding model not loaded.")

    temp_pdf_path = None
    try:
        # 1. Download and Process Document
        logging.info(f"Step 1: Downloading document from {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()
        logging.info("Document downloaded successfully.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            temp_pdf_path = tmp_file.name
        
        logging.info(f"Step 2: Loading and processing PDF from {temp_pdf_path}")
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logging.info(f"PDF processed into {len(chunks)} chunks.")
        
        logging.info("Step 3: Creating FAISS vector store from chunks.")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        logging.info("In-memory vector store and retriever created.")

        # 2. Process Each Question with the Multi-Agent Workflow
        answers = []
        for i, question in enumerate(questions):
            logging.info(f"--- Processing question {i+1}/{len(questions)}: '{question}' ---")
            
            # Step A: Call the Router Agent
            logging.info("Step 4a: Classifying question with router agent.")
            route = router_agent.invoke({"question": question})
            query_type = route.query_type
            logging.info(f"Router classified as: {query_type}")

            # Step B: Handle different query types
            if query_type == "fallback":
                answer = "I'm sorry, I can only answer questions about the provided document. Please ask a question relevant to the document's content."
                logging.info("Fallback triggered. Skipping RAG.")
            else:
                # For all other relevant queries, perform the RAG process
                logging.info("Step 4b: Retrieving relevant documents.")
                relevant_docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                logging.info(f"Retrieved {len(relevant_docs)} documents for context.")
                
                # Step C: Call the Synthesizer Agent
                logging.info("Step 4c: Synthesizing answer with synthesizer agent.")
                result = synthesizer_agent.invoke({
                    "context": context,
                    "question": question
                })
                
                answer = result.content.strip()
                logging.info("Answer synthesized successfully.")

            answers.append(answer)
            
        logging.info("--- All questions processed successfully. ---")
        return answers

    except Exception as e:
        logging.error(f"An exception occurred in process_rag_query: {e}")
        logging.error(traceback.format_exc())
        # Re-raise the exception to be caught by the main endpoint handler
        raise

    finally:
        # ... (cleanup code) ...
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logging.info(f"Cleaned up temporary file: {temp_pdf_path}")