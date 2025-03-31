from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
from app.services.candidate_service import CandidateService
from app.utils.retell import verify_retell_signature
from app.config import get_settings
import traceback
from typing import Dict, Any
import hmac
import hashlib

# Get logger
logger = logging.getLogger(__name__)

candidate_service = CandidateService()
settings = get_settings()

def extract_call_data(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate relevant call data from webhook payload."""
    logger.info("🔍 Extracting call data from payload")
    
    # Log the raw body structure
    logger.info("📦 Raw webhook payload structure:")
    for key, value in body.items():
        logger.info(f"  {key}: {type(value).__name__} = {value}")
    
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
        "access_token": call_data.get("access_token"),
        "retell_llm_dynamic_variables": call_data.get("retell_llm_dynamic_variables", {}),
        "opt_out_sensitive_data_storage": call_data.get("opt_out_sensitive_data_storage", False),
        "latency": call_data.get("latency", {}),
        "call_cost": call_data.get("call_cost", {})
    }
    
    # Determine event type based on call_status if not already set
    if not extracted["event"]:
        if extracted["call_status"] == "registered":
            extracted["event"] = "call_started"
        elif extracted["call_status"] == "ended":
            extracted["event"] = "call_ended"
        elif "call_analysis" in call_data:
            extracted["event"] = "call_analyzed"
    
    # Add call analysis details if available
    if extracted["call_analysis"]:
        extracted.update({
            "call_summary": extracted["call_analysis"].get("call_summary"),
            "user_sentiment": extracted["call_analysis"].get("user_sentiment"),
            "call_successful": extracted["call_analysis"].get("call_successful", False),
            "in_voicemail": extracted["call_analysis"].get("in_voicemail", False)
        })
    
    # Log what we found
    logger.info("📋 Extracted fields:")
    for key, value in extracted.items():
        if value is not None and value != {} and value != []:
            logger.info(f"  {key}: {value}")
    
    return extracted

async def handler(request):
    """Handle Retell webhook events."""
    try:
        # Log webhook data
        logger.info("📥 Received Retell webhook")
        logger.info(f"Request URL: {request.url}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Get request body
        body = await request.json()
        # Format payload exactly as Retell expects
        raw_body = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        logger.info(f"📦 Raw webhook payload: {raw_body}")
        
        # Detailed payload structure logging
        logger.info("🔍 Detailed Payload Analysis:")
        logger.info("📦 Webhook payload structure:")
        for key, value in body.items():
            logger.info(f"  {key}: {type(value).__name__} = {value}")
            if isinstance(value, dict):
                logger.info(f"    Nested structure for {key}:")
                for nested_key, nested_value in value.items():
                    logger.info(f"      {nested_key}: {type(nested_value).__name__} = {nested_value}")
            elif isinstance(value, list):
                logger.info(f"    List length for {key}: {len(value)}")
                if value and isinstance(value[0], dict):
                    logger.info(f"    First item structure in {key}:")
                    for nested_key, nested_value in value[0].items():
                        logger.info(f"      {nested_key}: {type(nested_value).__name__} = {nested_value}")

        # Verify Retell signature
        signature = request.headers.get("X-Retell-Signature")
        if not signature:
            logger.error("❌ Missing Retell signature header")
            return {
                "statusCode": 401,
                "body": json.dumps({
                    "error": "Missing X-Retell-Signature header"
                })
            }

        # Log signature verification details
        logger.info(f"🔑 Received signature: {signature}")
        logger.info(f"🔑 Using API key: {settings.retell_api_key[:5]}...")  # Only log first 5 chars for security

        is_valid = verify_retell_signature(
            payload=raw_body,
            signature=signature,
            api_key=settings.retell_api_key
        )

        if not is_valid:
            logger.error("❌ Invalid Retell signature")
            return {
                "statusCode": 401,
                "body": json.dumps({
                    "error": "Invalid signature"
                })
            }

        logger.info("✅ Retell signature verified")

        # Extract call data
        logger.info("🔍 Starting call data extraction...")
        call_data = extract_call_data(body)
        event = call_data["event"]
        call_id = call_data["call_id"]
        
        # Log extracted data
        logger.info("📞 Extracted call data:")
        logger.info(f"Event: {event}")
        logger.info(f"Call ID: {call_id}")
        logger.info(f"Call Status: {call_data['call_status']}")
        logger.info(f"Start Time: {datetime.fromtimestamp(call_data['start_timestamp']/1000) if call_data['start_timestamp'] else 'N/A'}")
        logger.info(f"End Time: {datetime.fromtimestamp(call_data['end_timestamp']/1000) if call_data['end_timestamp'] else 'N/A'}")
        
        # Log call analysis details if available
        if call_data.get('call_analysis'):
            logger.info("📊 Call Analysis Details:")
            logger.info(f"  Summary: {call_data['call_analysis'].get('call_summary', 'N/A')}")
            logger.info(f"  Sentiment: {call_data['call_analysis'].get('user_sentiment', 'N/A')}")
            logger.info(f"  Successful: {call_data['call_analysis'].get('call_successful', False)}")
            logger.info(f"  Voicemail: {call_data['call_analysis'].get('in_voicemail', False)}")
            
        # Log call cost if available
        if call_data.get('call_cost'):
            logger.info("💰 Call Cost Details:")
            logger.info(f"  Total Duration: {call_data['call_cost'].get('total_duration_seconds', 0)} seconds")
            logger.info(f"  Total Cost: ${call_data['call_cost'].get('combined_cost', 0)}")

        if not call_id:
            logger.error("❌ Missing call_id in webhook payload")
            logger.error("🔍 Payload keys available:")
            for key in body.keys():
                logger.error(f"  - {key}")
            logger.error("🔍 Full payload structure:")
            logger.error(json.dumps(body, indent=2))
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Missing required field: call_id"
                })
            }

        # Handle different event types
        if event == "call_started":
            logger.info(f"📞 Call started: {call_id}")
        elif event == "call_ended":
            logger.info(f"📞 Call ended: {call_id}")
            try:
                # Process the call completion
                await candidate_service.process_call_completion(call_data)
                logger.info(f"✅ Successfully processed call completion for call {call_id}")
            except Exception as e:
                logger.error(f"❌ Error processing call completion: {str(e)}")
                return {
                    "statusCode": 500,
                    "body": json.dumps({
                        "error": f"Error processing call completion: {str(e)}"
                    })
                }
        elif event == "call_analyzed":
            logger.info(f"📞 Call analyzed: {call_id}")
            logger.info(f"Call Summary: {call_data.get('call_summary', 'N/A')}")
            logger.info(f"User Sentiment: {call_data.get('user_sentiment', 'N/A')}")
            logger.info(f"Call Successful: {call_data.get('call_successful', False)}")
        else:
            logger.warning(f"⚠️ Unknown event type: {event}")

        # Return 204 No Content for successful webhook processing
        return {
            "statusCode": 204,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            },
            "body": ""  # Empty body for 204 response
        }

    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Error processing webhook: {str(e)}"
            })
        } 