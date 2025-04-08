import asyncio
import sys
from pathlib import Path
import json
import httpx
from typing import Dict, Any

# Add the app directory to the Python path
app_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(app_dir))

from app.config.settings import get_settings
from app.services.retell_service import RetellService

async def trigger_retrospective_webhook(call_id: str, candidate_id: str):
    """
    Trigger the Retell webhook retrospectively for a specific call.
    """
    try:
        # Initialize settings and services
        settings = get_settings()
        retell_service = RetellService(settings)
        
        # Get full call data from Retell
        print(f"Fetching call data for {call_id}...")
        call_data = await retell_service.get_call(call_id)
        if not call_data:
            print(f"❌ No call data found for call {call_id}")
            return
            
        # Format the webhook payload
        webhook_payload = {
            "call": call_data,
            "event": "call_analyzed",  # Set as analyzed to trigger processing
            "call_status": call_data.get("call_status", "ended"),
            "metadata": {
                "candidate_id": candidate_id
            }
        }
        
        # Send to webhook endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://anita-fastapi.onrender.com/webhook/retell",  # Adjust URL as needed
                json=webhook_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ Successfully triggered webhook for call {call_id}")
                print(f"Response: {response.json()}")
            else:
                print(f"❌ Failed to trigger webhook: {response.status_code} - {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    call_id = "call_88c308056d2b13f4bdade40b101"
    candidate_id = "f4bdc46d-e5f3-4b60-89af-7d2ebe0b2adf"
    
    asyncio.run(trigger_retrospective_webhook(call_id, candidate_id)) 