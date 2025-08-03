# main.py (Final Production-Ready Version)
import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import LangChain components with the correct, modern paths
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Load Environment Variables ---
load_dotenv()

# --- Pydantic Models for API ---
class Query(BaseModel):
    question: str

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]

# --- Initialize FastAPI App ---
app = FastAPI(
    title="HackRx RAG API",
    description="An API for the HackRx 6.0 hackathon to query documents using RAG.",
    version="3.0.0" # Final version
)

# --- Load Models and RAG Chain (This happens only once on startup) ---
qa_chain = None
try:
    print("--- STARTUP: Loading models and connecting to vector store... ---")

    # Initialize the Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # **THE FIX IS HERE**: We now connect to an EXISTING collection
    # This code will NOT try to create a new one.
    vector_store = AstraDBVectorStore(
        embedding=embedding_model,
        collection_name="my_rag_collection",
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    )
    print("--- STARTUP: Successfully connected to Astra DB collection. ---")

    # Initialize the LLM
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

    # Create the RAG Prompt Template
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

    # Create the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("--- STARTUP: Models and RAG chain loaded successfully. ---")
except Exception as e:
    print("\n--- FATAL STARTUP ERROR ---")
    traceback.print_exc()
    print("---------------------------\n")

# --- API Endpoint ---
@app.post("/ask", response_model=RAGResponse)
async def ask_question(query: Query):
    if not qa_chain:
        raise HTTPException(status_code=500, detail="RAG chain is not available due to a startup error.")

    try:
        result = qa_chain.invoke({"query": query.question})
        response_data = {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
        return response_data
    except Exception as e:
        print("\n--- EXCEPTION IN /ask ---")
        traceback.print_exc()
        print("-------------------------\n")
        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs.")

@app.get("/")
def read_root():
    return {"status": "API is online and ready."}
