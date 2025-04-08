from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
import traceback
from typing import Dict, Any
from fastapi import Request, BackgroundTasks, APIRouter, Depends
from fastapi.responses import JSONResponse
from app.agents.brain_agent import BrainAgent
from retell import Retell
from supabase._async.client import AsyncClient
from app.config.settings import Settings
from app.config.utils import get_table_name
from app.dependencies import get_vector_service, get_brain_agent, get_supabase_client_dependency, get_cached_settings, get_retell_service
from app.services.vector_service import VectorService
from app.services.retell_service import RetellService
import uuid
import httpx

# Get logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

async def log_call_communication(
    candidate_id: str,
    call_data: Dict[str, Any],
    supabase_client: AsyncClient,
    settings: Settings
):
    """Logs the completed call details to the communications table."""
    logger.info(f"Attempting to log call communication for candidate {candidate_id}")
    try:
        call_id = call_data.get('call_id')
        if not call_id:
            logger.warning(f"Missing call_id in call_data for candidate {candidate_id}. Cannot set thread_id.")
            thread_id_to_log = None
        else:
            # Generate a new UUID for thread_id while preserving the call_id in metadata
            thread_id_to_log = str(uuid.uuid4())

        # Extract transcript summary instead of full transcript
        transcript_object = call_data.get('transcript_object', [])
        
        # Safely calculate transcript summary with better error handling
        transcript_summary = {
            "word_count": len(transcript_object)
        }
        
        # Safely calculate duration only if transcript has valid structure
        if transcript_object and len(transcript_object) > 0:
            try:
                # Check if first and last items have required keys
                if "start" in transcript_object[0] and "end" in transcript_object[-1]:
                    transcript_summary["duration"] = transcript_object[-1]["end"] - transcript_object[0]["start"]
                else:
                    transcript_summary["duration"] = 0
                    logger.warning(f"Transcript missing start/end times for candidate {candidate_id}")
            except Exception as e:
                transcript_summary["duration"] = 0
                logger.warning(f"Error calculating transcript duration: {str(e)}")
        else:
            transcript_summary["duration"] = 0
        
        # Safely extract first few words
        first_words = ""
        if transcript_object:
            try:
                words = []
                for word_obj in transcript_object[:5]:
                    if "word" in word_obj:
                        words.append(word_obj["word"])
                first_words = " ".join(words) + "..." if words else ""
            except Exception as e:
                logger.warning(f"Error extracting first words from transcript: {str(e)}")
        transcript_summary["first_words"] = first_words

        # Create communication log record with required fields
        communication_log = {
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "candidates_id": candidate_id,
            "thread_id": thread_id_to_log if thread_id_to_log else str(uuid.uuid4()),  # Ensure we always have a thread_id
            "type": "call",
            "direction": "inbound",
            "subject": f"Retell Call ({call_id})" if call_id else "Retell Call",
            "content": json.dumps(transcript_summary),
            "metadata": {
                "call_id": call_id,
                "call_status": call_data.get('call_status'),
                "start_timestamp": call_data.get('start_timestamp'),
                "end_timestamp": call_data.get('end_timestamp'),
                "recording_url": call_data.get('recording_url'),
                "disconnection_reason": call_data.get('disconnection_reason'),
                "agent_id": call_data.get('agent_id'),
                "call_analysis": call_data.get('call_analysis', {}),
                "full_transcript": json.dumps(transcript_object if transcript_object else [])  # Ensure we don't serialize None
            },
            "timestamp": datetime.utcnow().isoformat()  # Explicitly set timestamp
        }
        
        logger.info(f"Preparing to insert communication log for candidate {candidate_id}")
        # Don't log the full communication_log object as it contains large transcript data
        
        table_name = get_table_name("communications", settings)
        logger.info(f"Using table name: {table_name}")
        
        log_resp = await supabase_client.table(table_name).insert(communication_log).execute()
        
        if hasattr(log_resp, 'data') and log_resp.data:
            logger.info(f"Successfully logged call communication for candidate {candidate_id}")
            # Don't log the response data which might contain sensitive information
        else:
            logger.warning(f"Could not log call communication for candidate {candidate_id}")
            if hasattr(log_resp, 'error'):
                logger.error(f"Supabase error: {log_resp.error}")
    except Exception as log_err:
        logger.error(f"Error logging call communication for candidate {candidate_id}: {log_err}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Try to log the error message that might help debug the issue
        if 'communication_log' in locals():
            # Create a copy without the large transcript data for logging
            log_copy = communication_log.copy()
            if 'metadata' in log_copy and 'full_transcript' in log_copy['metadata']:
                log_copy['metadata'] = log_copy['metadata'].copy()
                log_copy['metadata']['full_transcript'] = "[REDACTED]"
            # Don't log even the redacted version, just note there was an error
            logger.error(f"Error occurred with communication log object (transcript redacted)")

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
        "agent_id": call_data.get("agent_id"),
        "call_cost": call_data.get("call_cost", {})
    }
    
    # Log what we found, but skip transcript object and other large fields
    logger.info("📋 Extracted fields:")
    for key, value in extracted.items():
        if (value is not None and value != {} and value != [] and 
            key not in ['transcript_object', 'transcript', 'call_analysis', 'call_cost']):
            logger.info(f"  {key}: {value}")
    
    return extracted

@router.post("/retell")
async def handler(
    request: Request,
    background_tasks: BackgroundTasks,
    brain_agent: BrainAgent = Depends(get_brain_agent),
    supabase_client: AsyncClient = Depends(get_supabase_client_dependency),
    settings: Settings = Depends(get_cached_settings),
    retell_service: RetellService = Depends(get_retell_service)
):
    """Handle Retell webhook events."""
    try:
        payload = await request.json()
        
        # Log the incoming request details
        logger.info(f"🔔 Received Retell webhook at {request.url}")
        
        # Extract call data first to get environment from metadata
        call_data = extract_call_data(payload)
        if not call_data:
            logger.error("❌ Failed to extract call data from webhook payload")
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid webhook payload"})
            
        # Get originating environment from metadata
        webhook_env = call_data.get('metadata', {}).get('environment')
        
        logger.info(f"Environment settings:")
        logger.info(f"  - Originating ENV: {webhook_env}")
        logger.info(f"  - Current Server ENV: {settings.environment}")
        
        # Forward webhooks based on originating environment
        if settings.environment == "production":
            # Production server should forward to appropriate environment
            if webhook_env == "development":
                if not settings.development_webhook_url:
                    logger.error("❌ No development webhook URL configured")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Development webhook URL not configured"}
                    )
                forward_url = settings.development_webhook_url
                logger.info(f"⏩ Forwarding development webhook to {forward_url}")
            elif webhook_env == "staging":
                forward_url = "https://anita-fastapi-staging.onrender.com/webhook/retell"
                logger.info(f"⏩ Forwarding staging webhook to staging server")
            else:
                # Process in production
                forward_url = None
                logger.info("✅ Processing webhook in production")
                
            if forward_url:
                try:
                    async with httpx.AsyncClient() as client:
                        logger.info(f"Attempting to forward webhook to {forward_url}")
                        response = await client.post(
                            forward_url,
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=10.0  # Add timeout to prevent hanging
                        )
                        if response.status_code == 200:
                            logger.info(f"✅ Successfully forwarded webhook to {webhook_env}")
                            return JSONResponse(
                                status_code=200,
                                content={"status": "success", "message": f"Forwarded to {webhook_env}"}
                            )
                        else:
                            logger.error(f"❌ Error forwarding to {webhook_env}: {response.status_code} - {response.text}")
                            # Continue with production processing as fallback
                            logger.info("⚠️ Falling back to production processing")
                except Exception as e:
                    logger.error(f"❌ Failed to forward webhook: {str(e)}")
                    # Continue with production processing as fallback
                    logger.info("⚠️ Falling back to production processing")

        # Only process the webhook if:
        # 1. We're in production and it's a production webhook
        # 2. We're in staging and it's a staging webhook
        # 3. We're in development (localhost) and it's a development webhook
        should_process = (
            (settings.environment == webhook_env) or
            (settings.environment == "development" and "localhost" in str(request.url))
        )

        if not should_process:
            logger.info(f"⏭️ Skipping processing - Current env: {settings.environment}, Originating env: {webhook_env}")
            return JSONResponse(
                status_code=200, 
                content={"status": "success", "message": f"Skipped processing - Not meant for {settings.environment}"}
            )

        # Process the webhook
        # Get the call ID
        call_id = call_data.get('call_id')
        if not call_id:
            logger.error("❌ No call_id found in call data")
            return JSONResponse(status_code=400, content={"status": "error", "message": "No call_id in call data"})

        # Always fetch the full call data from Retell API
        try:
            logger.info(f"Fetching full call data for call {call_id}")
            full_call_data = await retell_service.get_call(call_id)
            if full_call_data:
                # Update call_data with full transcript and other data
                call_data['transcript'] = full_call_data.get('transcript', '')
                call_data['transcript_object'] = full_call_data.get('transcript_object', [])
                call_data['transcript_with_tool_calls'] = full_call_data.get('transcript_with_tool_calls', [])
                logger.info(f"Successfully fetched full call data for call {call_id}")
            else:
                logger.warning(f"No full call data available for call {call_id}")
        except Exception as e:
            logger.error(f"Error fetching full call data: {str(e)}")
            # Continue with the original call_data even if fetch fails

        # Add debug logging for transcript
        logger.debug(f"Transcript content: {call_data.get('transcript')}")
        logger.debug(f"Transcript object length: {len(call_data.get('transcript_object', []))}")
        if call_data.get('transcript_object'):
            logger.debug(f"First few transcript objects: {call_data.get('transcript_object')[:3]}")

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
            logger.info(f"📞 Call analyzed for candidate {candidate_id}. Logging communication and routing to BrainAgent.")
            # Add task to log the call communication FIRST
            background_tasks.add_task(
                log_call_communication,
                candidate_id=candidate_id,
                call_data=call_data,
                supabase_client=supabase_client,
                settings=settings
            )
            # Then add task for BrainAgent processing
            background_tasks.add_task(
                brain_agent.handle_call_processed,
                candidate_id=candidate_id,
                call_data=call_data
            )
            return JSONResponse(content={"status": "success", "message": "Call logged and processing delegated to BrainAgent"})
        else:
            # Log other events but don't trigger the main processing
            logger.info(f"Webhook event '{event_type}' (Status: '{call_status}') received but not the primary trigger ('call_analyzed'). No action taken.")
            return JSONResponse(content={"status": "success", "message": f"Webhook received, event '{event_type}' ignored"})

    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}\nTraceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)}) # Return 500 on unexpected errors 