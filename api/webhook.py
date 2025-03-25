from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(request):
    """Handle Retell webhook events."""
    try:
        # Log webhook data
        logger.info("📥 Received Retell webhook")
        logger.info(f"Headers: {dict(request.headers)}")
        
        # Get request body
        body = request.get_json()
        logger.info(f"📦 Webhook payload: {json.dumps(body, indent=2)}")

        # Extract event data
        event = body.get("event")
        call_data = body.get("call", {})
        call_id = call_data.get("call_id")
        call_status = call_data.get("call_status")
        metadata = call_data.get("metadata", {})

        if not event or not call_id:
            logger.error("❌ Missing required fields in webhook payload")
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Missing required fields: event or call_id"
                })
            }

        logger.info(f"📞 Processing {event} event for call {call_id}")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            },
            "body": json.dumps({
                "status": "success",
                "message": "Webhook received",
                "timestamp": datetime.now().isoformat()
            })
        }

    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Error processing webhook: {str(e)}"
            })
        } 