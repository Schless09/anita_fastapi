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
brain_agent = BrainAgent()

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
        payload = await request.json()
        logger.info(f"📥 Received Retell webhook: {json.dumps(payload, indent=2)}")

        call_data = extract_call_data(payload)
        if not call_data:
            logger.error("❌ Failed to extract call data from webhook payload")
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid webhook payload"})

        candidate_id = call_data.get('metadata', {}).get('candidate_id')
        if not candidate_id:
            logger.error("❌ No candidate_id found in call metadata")
            return JSONResponse(status_code=400, content={"status": "error", "message": "No candidate_id in metadata"})

        event_type = call_data.get('event')
        call_status = call_data.get('call_status')

        logger.info(f"📞 Processing webhook for candidate {candidate_id}")
        logger.info(f"Event type: {event_type}, Call status: {call_status}")

        # --- Trigger brain agent ONLY on the final analysis event --- 
        # Use 'call_analyzed' as the primary trigger, assuming it contains final data
        if event_type == 'call_analyzed':
            logger.info(f"🧠 Routing 'call_analyzed' event for candidate {candidate_id} to BrainAgent")
            background_tasks.add_task(
                brain_agent.handle_call_processed, # Target the correct method
                candidate_id=candidate_id,
                call_data=call_data
            )
            return JSONResponse(content={"status": "success", "message": "'call_analyzed' processing delegated to BrainAgent"})
        else:
            # Log other events but don't trigger the main processing
            logger.info(f"Webhook event '{event_type}' (Status: '{call_status}') received but not the primary trigger ('call_analyzed'). No action taken.")
            return JSONResponse(content={"status": "success", "message": f"Webhook received, event '{event_type}' ignored"})

    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}\nTraceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)}) # Return 500 on unexpected errors 