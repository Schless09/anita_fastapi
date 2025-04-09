"""Slack notification service for candidate and call updates."""
import logging
import ssl
import certifi # Import certifi
from typing import Dict, Any, List, Optional
from app.config.settings import Settings, get_settings
from slack_sdk.webhook import WebhookClient
from slack_sdk.webhook.webhook_response import WebhookResponse
import traceback

logger = logging.getLogger(__name__)

class SlackService:
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the Slack service.
        
        Args:
            settings: Optional Settings object. If not provided, will use get_settings()
        """
        self.settings = settings or get_settings()
        self.webhook_url = self.settings.slack_webhook_url
        
        # Create SSL context using certifi's CA bundle
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Initialize WebhookClient with the SSL context
        self.webhook_client = WebhookClient(
            self.webhook_url, 
            ssl=ssl_context # Pass the context here
        ) if self.webhook_url else None

    def test_connection(self) -> bool:
        """Sends a simple test message to the configured webhook URL."""
        if not self.webhook_client:
            logger.error("❌ Slack test failed: Webhook client not initialized (URL missing?)")
            return False
        try:
            response = self.webhook_client.send(text="Hello! This is a test message from Anita AI.")
            if response.status_code == 200:
                logger.info("✅ Slack connection test successful!")
                return True
            else:
                logger.error(f"❌ Slack connection test failed: {response.body}")
                return False
        except Exception as e:
            logger.error(f"❌ Slack connection test failed with exception: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}") # Add traceback for detail
            return False

    def _format_matches(self, matches: List[Dict[str, Any]]) -> str:
        """Format job matches into a readable string with percentages."""
        if not matches:
            return "_No relevant job matches found._"
        
        formatted_matches = []
        # Sort matches by similarity score, descending
        sorted_matches = sorted(matches, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for match in sorted_matches:
            # Assuming similarity is between 0 and 1
            score = match.get('similarity', 0) * 100  
            formatted_matches.append(
                f"• *{match.get('title', 'Unknown Role')}* at *{match.get('company', 'Unknown Company')}* "
                f"(Match: {score:.1f}%)"
            )
        return "\n".join(formatted_matches)

    async def notify_call_processed(
        self,
        candidate_name: str,
        email: str,
        phone: str,
        transcript: Optional[str],
        matches: Optional[List[Dict[str, Any]]] = None,
        linkedin_url: Optional[str] = None
    ) -> bool:
        """Send a notification after a call has been processed."""
        if not self.webhook_client:
            logger.warning("Slack webhook URL not configured. Skipping notification.")
            return False

        try:
            # Prepare fields for the main section
            info_fields = [
                {"type": "mrkdwn", "text": f"*Name:*\n{candidate_name}"},
                {"type": "mrkdwn", "text": f"*Email:*\n{email}"},
                {"type": "mrkdwn", "text": f"*Phone:*\n{phone or '_Not Provided_'}"}
            ]
            
            # Add LinkedIn URL field if available, ensuring protocol
            if linkedin_url:
                # Ensure URL starts with http:// or https://
                if not linkedin_url.startswith(("http://", "https://")):
                    processed_url = f"https://{linkedin_url}"
                else:
                    processed_url = linkedin_url
                # Format as clickable link using the processed URL
                info_fields.append({"type": "mrkdwn", "text": f"*LinkedIn:*\n<{processed_url}|Profile>"})
            
            # Construct blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "New Candidate! 🚀",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": info_fields 
                }
            ]

            # Add Matches Section
            if matches is not None: 
                blocks.extend([
                    {"type": "divider"},
                    {
                        "type": "section",
                        "text": {
                             "type": "mrkdwn",
                             "text": f"*Job Matches:*\n{self._format_matches(matches)}"
                         }
                    }
                ])
            
            # Add Transcript Section
            if transcript:
                max_length = 2900  
                truncated_transcript = transcript[:max_length] + "... (truncated)" if len(transcript) > max_length else transcript
                blocks.extend([
                    {"type": "divider"},
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Call Transcript:*\n```{truncated_transcript}```"
                        }
                    }
                ])
            else:
                 blocks.extend([
                    {"type": "divider"},
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Call Transcript:*\n_Transcript not available._"
                        }
                    }
                 ])

            # Send the message using the webhook client
            response: WebhookResponse = self.webhook_client.send(
                text=f"Call Processed for {candidate_name}",  # Fallback text
                blocks=blocks
            )

            if response.status_code == 200:
                logger.info(f"✅ Successfully sent Slack notification for call processed: {email}")
                return True
            else:
                logger.error(f"❌ Failed to send Slack notification (Call Processed): {response.body} for {email}")
                return False

        except Exception as e:
            logger.error(f"❌ Error sending Slack notification (Call Processed): {str(e)} for {email}")
            return False 