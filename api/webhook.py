from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/webhook")
async def webhook_handler(request: Request):
    """Handle Retell webhook events."""
    try:
        # Log webhook data
        logger.info("📥 Received Retell webhook")
        logger.info(f"Headers: {dict(request.headers)}")
        
        # Get request body
        body = await request.json()
        logger.info(f"📦 Webhook payload: {json.dumps(body, indent=2)}")

        # Extract event data
        event = body.get("event")
        call_data = body.get("call", {})
        call_id = call_data.get("call_id")
        call_status = call_data.get("call_status")
        metadata = call_data.get("metadata", {})

        if not event or not call_id:
            logger.error("❌ Missing required fields in webhook payload")
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: event or call_id"
            )

        logger.info(f"📞 Processing {event} event for call {call_id}")

        return {
            "status": "success",
            "message": "Webhook received",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing webhook: {str(e)}"
        ) 