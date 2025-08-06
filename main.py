import os
import traceback
import requests
import tempfile
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# Import modern LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Load Environment Variables ---
load_dotenv()

# --- Pydantic Models for the Hackathon API Format ---
class HackathonRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")


# --- Initialize FastAPI App ---
app = FastAPI(
    title="High-Accuracy HackRx RAG API",
    description="An advanced RAG API for the HackRx 6.0 hackathon with token authentication.",
    version="5.0.0" # Final submission version
)

# --- Load Foundational Models (This happens only once on startup) ---
try:
    print("--- STARTUP: Loading foundational models... ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0) # Low temp for high accuracy
    print("--- STARTUP: Foundational models loaded successfully. ---")
except Exception as e:
    print("\n--- FATAL STARTUP ERROR ---")
    traceback.print_exc()
    embedding_model = None
    llm = None

# --- Advanced Prompt Template for High Accuracy ---
QA_SYSTEM_PROMPT = """You are a claims adjudicator for an insurance company. Your task is to determine coverage based STRICTLY on the provided policy document.

*INSTRUCTIONS:*
1.  Your primary goal is to determine if a claim is covered or excluded.
2.  Pay extremely close attention to waiting periods, exclusions, and conditions. If a claim falls within a waiting period or is listed as an exclusion, it is NOT covered.
3.  Provide a clear "Yes" or "No" answer, followed by a brief explanation based on the exact wording of the policy.
4.  Quote the relevant section of the policy to support your answer.
5.  If the information required to answer the question is not found in the provided context, you MUST respond with the exact phrase: "Information not found in the documents."

Context:
{context}

Question:
{input}

Answer:"""
qa_prompt = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT)

# --- API Endpoint as specified by the Hackathon ---
@app.post("/hackrx/run")
async def run_qa(
    req: HackathonRequest,
    authorization: Optional[str] = Header(None)
):
    # --- 1. Bearer Token Authentication ---
    expected_token = "6e6de8c174e72f2501628ae7ddc119732bc8c34a72097f682a2bf339db673dd7"
    expected_header = f"Bearer {expected_token}"
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization != expected_header:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    print("✅ Token validation passed.")

    if not embedding_model or not llm:
        raise HTTPException(status_code=500, detail="Foundational models are not available due to a startup error.")

    temp_pdf_path = None
    try:
        # --- 2. Download and Process the Document ---
        print(f"Downloading document from: {req.documents}")
        response = requests.get(req.documents)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name
        
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, embedding_model)
        print("In-memory vector store created for the request.")

        # --- 3. Create the Modern RAG Chain (LCEL) ---
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # --- 4. Process Each Question Individually ---
        answers = []
        print("\nProcessing questions...")
        for i, question in enumerate(req.questions):
            print(f"  - Answering question {i+1}: '{question}'")
            try:
                result = rag_chain.invoke({"input": question})
                answer_str = result.get("answer", "Information not found in the documents.")
                
                # Clean up the answer formatting
                cleaned_answer = answer_str.strip()
                answers.append(cleaned_answer)
            except Exception as e:
                print(f"    -> Error answering question {i+1}: {e}")
                answers.append("An error occurred while processing this question.")
        
        print("All questions processed.\n")

        separator = "\n\n" + "="*50 + "\n\n"
        return PlainTextResponse(content=separator.join(answers))

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or access the document URL: {e}")
    except Exception as e:
        print("\n--- EXCEPTION IN /hackrx/run ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    finally:
        # --- 5. Clean up the temporary file ---
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"Cleaned up temporary file: {temp_pdf_path}")

@app.get("/")
def health_check():
    return {"status": "ok"}
