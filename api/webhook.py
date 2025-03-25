from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Get request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            payload = json.loads(body)

            # Log webhook data
            logger.info("📥 Received Retell webhook")
            logger.info(f"Headers: {dict(self.headers)}")
            logger.info(f"📦 Webhook payload: {json.dumps(payload, indent=2)}")

            # Extract event data
            event = payload.get("event")
            call_data = payload.get("call", {})
            call_id = call_data.get("call_id")
            call_status = call_data.get("call_status")
            metadata = call_data.get("metadata", {})

            if not event or not call_id:
                self.send_error(400, "Missing required fields: event or call_id")
                return

            logger.info(f"📞 Processing {event} event for call {call_id}")

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()

            response = {
                "status": "success",
                "message": "Webhook received",
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            logger.error(f"❌ Error processing webhook: {str(e)}")
            self.send_error(500, f"Error processing webhook: {str(e)}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers() 