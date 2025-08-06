# app/agents.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

# Initialize a single, powerful LLM to be used by all agents
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

# --- 1. The Router Agent ---
class QueryClassifier(BaseModel):
    """Categorize the user's question."""
    query_type: Literal[
        "waiting_period",
        "coverage_limit",
        "exclusion_clause",
        "definition_of_terms",
        "general_query"
    ] = Field(description="The category of the user's insurance policy question.")

def get_router_agent():
    """Creates the agent responsible for classifying questions."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at analyzing insurance questions. Classify the user's query into one of the predefined categories."),
            ("human", "{question}")
        ]
    )
    # This creates a chain that forces the LLM to output structured JSON
    return prompt | llm.with_structured_output(QueryClassifier)

# --- 2. The Synthesizer Agent ---
def get_synthesizer_agent():
    """Creates the agent responsible for generating the final answer."""
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
{question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT)
    return prompt | llm