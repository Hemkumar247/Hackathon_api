#  This file handles all security-related logic, like validating the Bearer Token.
import os
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

# Load environment variables from your .env file
load_dotenv()

# Initialize the security scheme
http_bearer = HTTPBearer()

# Load the required token securely from the environment
EXPECTED_TOKEN = os.getenv("HACKATHON_TOKEN")

def validate_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer)):
    """
    Validates the Bearer Token from the Authorization header.
    """
    if not EXPECTED_TOKEN:
        # This is an important server-side check
        raise HTTPException(status_code=500, detail="Security token not configured on the server.")

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )
    
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token or authorization scheme",
        )
    
    print("✅ Token validation passed.")
    return True