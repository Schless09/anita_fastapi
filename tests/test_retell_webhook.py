import json
import hmac
import hashlib
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_retell_signature(payload: str, api_key: str) -> str:
    """Generate a Retell signature for testing."""
    return hmac.new(
        api_key.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

async def test_retell_webhook():
    """Test the Retell webhook endpoint."""
    # Get API key
    api_key = os.getenv('RETELL_API_KEY')
    if not api_key:
        print("❌ RETELL_API_KEY not found in environment variables")
        return

    # Test payload
    payload = {
        "event": "call_ended",
        "call": {
            "call_id": "test_call_123",
            "call_status": "ended",
            "transcript": "This is a test transcript",
            "metadata": {
                "candidate_id": "test_candidate_123"
            }
        }
    }

    # Convert payload to string with proper formatting
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    
    # Generate signature
    signature = generate_retell_signature(payload_str, api_key)

    # Headers
    headers = {
        "Content-Type": "application/json",
        "X-Retell-Signature": signature
    }

    # Send request
    async with httpx.AsyncClient() as client:
        try:
            print("📤 Sending test webhook request...")
            print(f"📦 Payload: {payload_str}")
            print(f"🔑 Signature: {signature}")
            
            response = await client.post(
                "http://localhost:8000/webhook/retell",
                json=payload,
                headers=headers
            )
            
            print(f"\n📥 Response status: {response.status_code}")
            print(f"📥 Response body: {response.text}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_retell_webhook()) 