#  This file handles all security-related logic, like validating the Bearer Token.

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

# Initialize the security scheme
http_bearer = HTTPBearer()

# This is the required token for the hackathon
EXPECTED_TOKEN = "6e6de8c174e72f2501628ae7ddc119732bc8c34a72097f682a2bf339db673dd7"

def validate_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer)):
    """
    Validates the Bearer Token from the Authorization header.
    """
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