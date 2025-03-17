from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

SENDGRID_CONFIG: Dict[str, Any] = {
    'api_key': os.getenv('SENDGRID_API_KEY'),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'inbound_parse': {
        'hostname': os.getenv('SENDGRID_INBOUND_HOSTNAME', 'your-domain.com'),
        'url': '/email/webhook',
        'spam_check': True,
        'send_raw': True
    }
}

def get_webhook_url() -> str:
    """Get the full webhook URL for SendGrid Inbound Parse setup."""
    hostname = SENDGRID_CONFIG['inbound_parse']['hostname']
    url_path = SENDGRID_CONFIG['inbound_parse']['url']
    return f"https://{hostname}{url_path}" 