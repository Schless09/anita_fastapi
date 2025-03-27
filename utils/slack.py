import os
import logging
import aiohttp
from typing import Optional
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

async def send_slack_notification(message: str, channel: Optional[str] = None) -> bool:
    """
    Send a notification to Slack.
    
    Args:
        message (str): The message to send
        channel (Optional[str]): The channel to send to. If None, uses default webhook channel
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            logger.error("❌ SLACK_WEBHOOK_URL not configured")
            return False
            
        payload = {"text": message}
        if channel:
            payload["channel"] = channel
            
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("✅ Successfully sent Slack notification")
                    return True
                else:
                    logger.error(f"❌ Failed to send Slack notification. Status: {response.status}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Error sending Slack notification: {str(e)}")
        return False 