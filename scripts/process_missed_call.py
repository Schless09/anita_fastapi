import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(app_dir))

from app.services.candidate_service import CandidateService
from app.services.retell_service import RetellService

async def process_missed_call(candidate_id: str, call_id: str):
    """
    Manually process a missed call for a candidate.
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
            
        # Add candidate_id to metadata if not present
        if not call_data.get('metadata'):
            call_data['metadata'] = {}
        call_data['metadata']['candidate_id'] = candidate_id
        
        # Process the call
        await candidate_service.process_call_completion(call_data)
        print(f"✅ Successfully processed call for candidate {candidate_id}")
        
    except Exception as e:
        print(f"❌ Error processing call: {str(e)}")
        raise

if __name__ == "__main__":
    # Get candidate ID and call ID from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python process_missed_call.py <candidate_id> <call_id>")
        sys.exit(1)
        
    candidate_id = sys.argv[1]
    call_id = sys.argv[2]
    
    # Run the async function
    asyncio.run(process_missed_call(candidate_id, call_id)) 