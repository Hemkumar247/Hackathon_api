#  This file defines the structure of your API's JSON requests and responses.

from pydantic import BaseModel, Field
from typing import List

class HackathonRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="A list of questions to answer based on the document.")

class HackathonResponse(BaseModel):
    answers: List[str]
