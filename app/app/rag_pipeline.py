#  This is the core of your project. It contains all the logic for the RAG
#  pipeline, from downloading the document to generating the final answers.


import os
import traceback
import requests
import tempfile
from typing import List

# Import modern LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Load Foundational Models (This happens only once on startup) ---
try:
    print("--- RAG Pipeline: Loading foundational models... ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)
    print("--- RAG Pipeline: Foundational models loaded successfully. ---")
except Exception:
    print("\n--- RAG Pipeline: FATAL STARTUP ERROR ---")
    traceback.print_exc()
    embedding_model = None
    llm = None

# --- Advanced Prompt Template for High Accuracy ---
QA_SYSTEM_PROMPT = """You are an expert insurance policy analyst. Your task is to provide accurate, specific answers based STRICTLY on the provided context.

*INSTRUCTIONS:*
1.  Provide precise, factual responses without any speculation.
2.  Use the exact terminology found in the policy document.
3.  Include exact monetary amounts, percentages, and time periods exactly as they are stated in the context.
4.  Your response must be a single, complete sentence.
5.  If the information required to answer the question is not found in the provided context, you MUST respond with the exact phrase: "Information not found in the documents."

Context:
{context}

Question:
{input}

Answer:"""
qa_prompt = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT)


def process_rag_query(document_url: str, questions: List[str]) -> List[str]:
    """
    Main function to handle the entire RAG process for a given request.
    """
    if not embedding_model or not llm:
        raise RuntimeError("Foundational models are not loaded. Cannot process request.")

    temp_pdf_path = None
    try:
        # 1. Download and Process the Document
        print(f"Downloading document from: {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name
        
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, embedding_model)
        print("In-memory vector store created for the request.")

        # 2. Create the Modern RAG Chain (LCEL)
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # 3. Process Each Question Individually
        answers = []
        print("\nProcessing questions...")
        for i, question in enumerate(questions):
            print(f"  - Answering question {i+1}: '{question}'")
            try:
                result = rag_chain.invoke({"input": question})
                answer_str = result.get("answer", "Information not found in the documents.")
                
                # Clean up the answer formatting
                cleaned_answer = answer_str.strip()
                if cleaned_answer.startswith('"') and cleaned_answer.endswith('"'):
                    cleaned_answer = cleaned_answer[1:-1]
                
                answers.append(cleaned_answer)
            except Exception as e:
                print(f"    -> Error answering question {i+1}: {e}")
                answers.append("An error occurred while processing this question.")
        
        print("All questions processed.\n")
        return answers

    finally:
        # 4. Clean up the temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"Cleaned up temporary file: {temp_pdf_path}")
