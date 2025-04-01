import json
from retell import Retell
from app.config import get_settings

settings = get_settings()
retell = Retell(api_key=settings.retell_api_key)

def verify_retell_signature(
    payload: str,
    signature: str,
    api_key: str
) -> bool:
    """
    Verify a Retell webhook signature using their official SDK.
    
    Args:
        payload: The raw JSON payload as a string
        signature: The signature from X-Retell-Signature header
        api_key: The Retell API key
    
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        return retell.verify(
            payload,
            api_key=api_key,
            signature=signature
        )
    except Exception as e:
        print(f"Error verifying signature: {str(e)}")
        return False 