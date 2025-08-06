#  This file now only handles the API server and routes. It imports all the
#  heavy logic from the other files.
#
# ==============================================================================

from fastapi import FastAPI, HTTPException, Depends
import traceback
# Import the different parts of our application
from .schemas import HackathonRequest, HackathonResponse
from .security import validate_token
from .rag_pipeline import process_rag_query
from datetime import datetime
import logging
# --- Initialize FastAPI App ---
# Log to the console, which is standard for deployment platforms
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


app = FastAPI(
    title="High-Accuracy HackRx RAG API",
    description="A modular and advanced RAG API for the HackRx 6.0 hackathon.",
    version="5.0.0"
)

# --- API Endpoint as specified by the Hackathon ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def run_qa(
    req: HackathonRequest,
    is_authenticated: bool = Depends(validate_token)
):
    """
    This endpoint is secured by a Bearer Token. It accepts a document URL
    and a list of questions, then returns AI-generated answers.
    """
    try:
        # Call the main processing function from our RAG pipeline
        answers = process_rag_query(req.documents, req.questions)

        log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "documents": req.documents,
        "questions": req.questions,
        }
        logging.info(f"Received request: {log_entry}")
        return HackathonResponse(answers=answers)
        
    except Exception as e:
        print("\n--- EXCEPTION IN /hackrx/run ENDPOINT ---")
        traceback.print_exc()
        print("-----------------------------------------\n")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    

@app.get("/")
def health_check():
    return {"status": "ok"}