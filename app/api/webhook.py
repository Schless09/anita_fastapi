from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
from app.services.candidate_service import CandidateService
from app.config import get_settings
import traceback
from typing import Dict, Any
from fastapi import Request, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse
from app.agents.brain_agent import BrainAgent
from app.services.vector_service import VectorService
from retell import Retell

# Get logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services
candidate_service = CandidateService()
settings = get_settings()
vector_service = VectorService()
brain_agent = BrainAgent(vector_store=vector_service)

# Initialize Retell client
retell = Retell(api_key=str(settings.retell_api_key))

def extract_call_data(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate relevant call data from webhook payload."""
    logger.info("🔍 Extracting call data from payload")
    
    # Get call data from either root or nested call object
    call_data = body.get("call", body)  # If no call object, use body itself
    
    # Initialize with data from call object
    extracted = {
        "event": body.get("event"),  # Get event from root level
        "call_id": call_data.get("call_id"),
        "call_status": call_data.get("call_status"),
        "metadata": call_data.get("metadata", {}),
        "transcript": call_data.get("transcript"),
        "transcript_object": call_data.get("transcript_object", []),
        "call_analysis": call_data.get("call_analysis", {}),
        "recording_url": call_data.get("recording_url"),
        "start_timestamp": call_data.get("start_timestamp"),
        "end_timestamp": call_data.get("end_timestamp"),
        "disconnection_reason": call_data.get("disconnection_reason"),
        "call_type": call_data.get("call_type"),
        "agent_id": call_data.get("agent_id")
    }
    
    # Log what we found
    logger.info("📋 Extracted fields:")
    for key, value in extracted.items():
        if value is not None and value != {} and value != [] and key != 'transcript_object':  # Skip logging transcript_object
            logger.info(f"  {key}: {value}")
    
    return extracted

@router.post("/retell")
async def handler(request: Request, background_tasks: BackgroundTasks):
    """Handle Retell webhook events."""
    try:
        # Get raw body and parse JSON
        body = await request.body()
        post_data = json.loads(body)
        
        # Get signature from header
        signature = request.headers.get("x-retell-signature")
        if not signature:
            logger.error("❌ Missing Retell signature")
            return JSONResponse(
                status_code=401,
                content={"error": "Missing signature"}
            )
        
        # Verify signature using Retell SDK
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=str(settings.retell_api_key),
            signature=str(signature)
        )
        
        if not valid_signature:
            logger.error("❌ Invalid Retell signature")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid signature"}
            )
        
        # Extract essential call data
        call_data = extract_call_data(post_data)
        call_id = call_data.get('call_id')
        candidate_id = call_data.get('metadata', {}).get('candidate_id')
        call_status = call_data.get('call_status')
        
        # Log only essential information
        logger.info(f"📞 Call {call_id} - Status: {call_status}")
        
        # Skip processing if call is not completed
        if call_status not in ['ended', 'completed']:
            logger.info("⏳ Call not completed yet, skipping processing")
            return JSONResponse(
                status_code=200,
                content={"message": "Call not completed"}
            )
        
        # Process call completion
        try:
            await candidate_service.process_call_completion(call_data)
            
            # Check if profile is complete and trigger matching
            response = await candidate_service.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            profile = response.data.get('profile_json', {})
            
            if profile.get('processing_status', {}).get('resume_processed') and profile.get('processing_status', {}).get('call_completed'):
                logger.info("🎯 Profile complete, triggering matching")
                background_tasks.add_task(brain_agent.trigger_matching, candidate_id)
            
            return JSONResponse(
                status_code=200,
                content={"message": "Call processed successfully"}
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing call: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
            
    except Exception as e:
        logger.error(f"❌ Webhook error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        ) 