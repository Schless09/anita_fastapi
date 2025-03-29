import hmac
import hashlib
import json
from typing import Dict, Any

def verify_retell_signature(
    payload: str,
    signature: str,
    api_key: str
) -> bool:
    """
    Verify a Retell webhook signature.
    
    Args:
        payload: The raw JSON payload as a string
        signature: The signature from X-Retell-Signature header
        api_key: The Retell API key
    
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Create HMAC SHA-256 hash
        expected_signature = hmac.new(
            api_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False 