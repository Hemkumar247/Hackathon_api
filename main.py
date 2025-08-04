import os
import traceback
import requests
import tempfile
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import LangChain components with the correct, modern paths
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Using a fast, in-memory vector store
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Load Environment Variables ---
load_dotenv()

# --- Pydantic Models for the Hackathon API Format ---
class HackathonRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class HackathonResponse(BaseModel):
    answers: List[str]

# --- Initialize FastAPI App ---
app = FastAPI(
    title="High-Accuracy HackRx RAG API",
    description="An advanced RAG API for the HackRx 6.0 hackathon.",
    version="4.0.0" # Final version
)

# --- Load Foundational Models (This happens only once on startup) ---
# We load the embedding model and LLM once to be reused for every request.
try:
    print("--- STARTUP: Loading foundational models... ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1) # Low temperature for factual answers
    print("--- STARTUP: Foundational models loaded successfully. ---")
except Exception as e:
    print("\n--- FATAL STARTUP ERROR ---")
    traceback.print_exc()
    embedding_model = None
    llm = None

# --- Custom Prompt Template for High Accuracy ---
# This prompt is engineered to force the LLM to be concise and stick to the facts.
prompt_template = """
INSTRUCTIONS:
You are a highly intelligent insurance policy analysis assistant. Your task is to answer the user's question with extreme precision, based *only* on the context provided below.
- Provide a direct and concise answer.
- Do not add any conversational filler like "Based on the document...".
- If the context does not contain the information to answer the question, you MUST state: "The provided document does not contain enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{question}

PRECISE ANSWER:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Core RAG Logic ---
def answer_questions_from_url(doc_url: str, questions: List[str]) -> List[str]:
    """
    Downloads a PDF from a URL, processes it, and answers questions using a RAG pipeline.
    This is the core synchronous logic that can be shared between the API and other scripts.
    """
    if not embedding_model or not llm:
        raise RuntimeError("Foundational models are not available. Please check server startup logs.")

    temp_pdf_path = None
    try:
        # --- 1. Download the document from the URL ---
        print(f"Downloading document from: {doc_url}")
        response = requests.get(doc_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name
        print(f"Document downloaded and saved to: {temp_pdf_path}")

        # --- 2. Load and Process the Document ---
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, embedding_model)
        print("In-memory vector store created for the request.")

        # --- 3. Create the RAG Chain ---
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
            chain_type_kwargs={"prompt": PROMPT},
        )

        # --- 4. Process Each Question ---
        answers = []
        print("\nProcessing questions...")
        for i, question in enumerate(questions):
            print(f"  - Answering question {i+1}: '{question}'")
            try:
                result = qa_chain.invoke({"query": question})
                answers.append(result["result"].strip())
            except Exception as e:
                print(f"    -> Error answering question {i+1}: {e}")
                answers.append("An error occurred while processing this question.")
        
        print("All questions processed.\n")
        return answers

    finally:
        # --- 5. Clean up the temporary file ---
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"Cleaned up temporary file: {temp_pdf_path}")

# --- API Endpoint as specified by the Hackathon ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_documents_and_questions(request: HackathonRequest):
    try:
        answers = answer_questions_from_url(request.documents, request.questions)
        return {"answers": answers}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or access the document URL: {e}")
    except RuntimeError as e:
        # This catches the "Foundational models not available" error
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print("\n--- EXCEPTION IN /hackrx/run ---")
        traceback.print_exc()
        print("------------------------------\n")
        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs.")

@app.get("/")
def read_root():
    return {"status": "High-Accuracy API is online and ready."}
