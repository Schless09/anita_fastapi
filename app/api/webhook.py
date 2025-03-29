from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
from app.services.candidate_service import CandidateService
from app.utils.retell import verify_retell_signature
from app.config import get_settings
import traceback
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

candidate_service = CandidateService()
settings = get_settings()

def extract_call_data(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate relevant call data from webhook payload."""
    data = body.get("data", {})
    call_data = {
        "event": body.get("event"),
        "call_id": data.get("call_id"),
        "call_status": data.get("call_status"),
        "metadata": data.get("metadata", {}),
        "transcript": data.get("transcript"),
        "transcript_object": data.get("transcript_object", []),
        "call_analysis": data.get("call_analysis", {}),
        "recording_url": data.get("recording_url"),
        "start_timestamp": data.get("start_timestamp"),
        "end_timestamp": data.get("end_timestamp"),
        "disconnection_reason": data.get("disconnection_reason")
    }
    
    # Add call analysis details if available
    if call_data["call_analysis"]:
        call_data.update({
            "call_summary": call_data["call_analysis"].get("call_summary"),
            "user_sentiment": call_data["call_analysis"].get("user_sentiment"),
            "call_successful": call_data["call_analysis"].get("call_successful", False)
        })
    
    return call_data

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
        raw_body = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        logger.info(f"📦 Webhook payload: {raw_body}")

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

        if not call_id:
            logger.error("❌ Missing call_id in webhook payload")
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