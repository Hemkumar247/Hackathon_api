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

# --- Initialize the Agents ---
router_agent = get_router_agent()
synthesizer_agent = get_synthesizer_agent()


def process_rag_query(document_url: str, questions: List[str]) -> List[str]:
    if not embedding_model:
        raise RuntimeError("Embedding model not loaded.")

    temp_pdf_path = None
    try:
        # 1. Download and Process Document
        response = requests.get(document_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            temp_pdf_path = tmp_file.name

        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        print("In-memory vector store and retriever created.")

        # 2. Process Each Question with the Multi-Agent Workflow
        answers = []
        for question in questions:
            print(f"  - Answering question: '{question}'")
            
            # Step A: Call the Router Agent
            route = router_agent.invoke({"question": question})
            query_type = route.query_type
            print(f"    -> Router classified as: {query_type}")

            # Step B: Handle different query types
            if query_type == "fallback":
                answer = "I'm sorry, I can only answer questions about the provided document. Please ask a question relevant to the document's content."
            else:
                # For all other relevant queries, perform the RAG process
                relevant_docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Step C: Call the Synthesizer Agent
                result = synthesizer_agent.invoke({
                    "context": context,
                    "question": question
                })
                
                answer = result.content.strip()

            answers.append(answer)
            
        return answers

    finally:
        # ... (cleanup code) ...
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"Cleaned up temporary file: {temp_pdf_path}")