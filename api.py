import os
import tempfile
import requests
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import LangChain components
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load Environment Variables ---
load_dotenv()

# --- Constants and Configuration ---
EXPECTED_TOKEN = os.getenv("API_BEARER_TOKEN", "6e6de8c174e72f2501628ae7ddc119732bc8c34a72097f682a2bf339db673dd7")

# --- Pydantic Models for API ---
class HackRxRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize FastAPI App ---
app = FastAPI(
    title="HackRx Dynamic RAG API",
    description="An API to process a document from a URL and answer questions about it.",
    version="1.0.0"
)

# --- Security Dependency ---
async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme.")
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token.")
    return token

# --- Core RAG Processing Function ---
def process_document_and_answer(document_url: str, questions: List[str]) -> List[str]:
    try:
        # 1. Download the document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            response = requests.get(document_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # 2. Load the document
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # 3. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 4. Create an in-memory vector store
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embedding_model)

        # 5. Initialize the LLM and QA Chain
        llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        prompt_template = """
        INSTRUCTIONS:
        You are a helpful assistant. Your task is to answer the user's question based *only* on the context provided below.
        If the context does not contain the information needed to answer the question, you must state: "I do not have enough information in the provided document to answer this question."
        Do not make up information or use any external knowledge.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # 6. Process all questions
        answers = []
        for question in questions:
            result = qa_chain.invoke({"query": question})
            answers.append(result["result"])

        return answers

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        # Log the full error for debugging
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during document processing.")
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_hackrx(request: HackRxRequest):
    answers = process_document_and_answer(request.documents, request.questions)
    return {"answers": answers}

@app.get("/")
def read_root():
    return {"status": "HackRx Dynamic RAG API is online."}
