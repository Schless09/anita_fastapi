import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(app_dir))

from app.services.candidate_service import CandidateService
from app.services.retell_service import RetellService

async def process_missed_call(call_id: str):
    """
    Manually process a missed call.
    """
    try:
        # Initialize services
        retell_service = RetellService()
        candidate_service = CandidateService()
        
        # Get call data from Retell
        call_data = await retell_service.get_call(call_id)
        if not call_data:
            print(f"❌ No call data found for call {call_id}")
            return
            
        # Process the call
        await candidate_service.process_call_completion(call_data)
        print(f"✅ Successfully processed call {call_id}")
        
    except Exception as e:
        print(f"❌ Error processing call: {str(e)}")
        raise

if __name__ == "__main__":
    # Get call ID from command line argument
    if len(sys.argv) != 2:
        print("Usage: python process_missed_call.py <call_id>")
        sys.exit(1)
        
    call_id = sys.argv[1]
    
    # Run the async function
    asyncio.run(process_missed_call(call_id)) 