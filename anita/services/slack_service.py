import os
from slack_sdk.webhook import WebhookClient
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from app.config.settings import Settings
from loguru import logger
from typing import Tuple

class SlackService:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Prefer bot token for more capabilities, fall back to webhook
        if settings.slack_bot_token:
            self.client = WebClient(token=settings.slack_bot_token)
            logger.info("SlackService initialized with Bot Token.")
        elif settings.slack_webhook_url:
            self.client = WebhookClient(url=settings.slack_webhook_url)
            logger.info("SlackService initialized with Webhook URL.")
        else:
            self.client = None
            logger.warning("SlackService initialized without token or webhook URL. Notifications disabled.")
            
    def send_notification(self, channel_id: str, text: str, blocks: list = None):
        if not self.client:
            logger.warning("Slack client not configured. Cannot send notification.")
            return False
            
        try:
            if isinstance(self.client, WebClient):
                # Use chat_postMessage for bot tokens
                response = self.client.chat_postMessage(
                    channel=channel_id,
                    text=text,
                    blocks=blocks # Optional: Use block kit for richer messages
                )
                logger.info(f"Slack notification sent to channel {channel_id}.")
                return response.get("ok", False)
            elif isinstance(self.client, WebhookClient):
                # Use webhook send for webhook URLs
                response = self.client.send(
                    text=text,
                    blocks=blocks
                )
                logger.info(f"Slack notification sent via webhook.")
                # WebhookClient doesn't return 'ok' in the same way, check status code
                return response.status_code == 200
        except SlackApiError as e:
            logger.error(f"Error sending Slack notification to channel {channel_id}: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Slack notification: {e}")
            return False

    def format_reply_notification(self, candidate_email: str, candidate_message: str, ai_reply: str, inbound_comm_log_id: int) -> Tuple[str, list]:
        """Formats the notification message for Slack with action buttons."""
        
        # Store context needed for actions in the button values (or use private_metadata with modals)
        # Using inbound_comm_log_id is cleaner
        approve_action_id = "approve_email_reply"
        reject_action_id = "reject_email_reply"
        edit_action_id = "edit_email_reply"
        
        fallback_text = (
            f"New Candidate Reply from: {candidate_email}\n\n"
            f"*Candidate Message:*\n{candidate_message}\n\n"
            f"*Proposed AI Reply:*\n{ai_reply}\n\n"
            f"(Log ID: {inbound_comm_log_id})"
        )

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📧 New Candidate Email Reply",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*From:* {candidate_email}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Log ID:* {inbound_comm_log_id}" # Display log ID for reference
                    }
                ]
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Candidate's Message:*\n```" + candidate_message + "```"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Proposed AI Reply:*\n```" + ai_reply + "```"
                }
            },
            {
                 "type": "actions",
                 "block_id": f"email_actions_{inbound_comm_log_id}", # Unique block ID
                 "elements": [
                     {
                         "type": "button",
                         "text": {
                             "type": "plain_text",
                             "text": "Approve & Send",
                             "emoji": True
                         },
                         "style": "primary",
                         "value": str(inbound_comm_log_id), # Pass log ID
                         "action_id": approve_action_id
                     },
                     {
                         "type": "button",
                         "text": {
                             "type": "plain_text",
                             "text": "Edit Reply",
                             "emoji": True
                         },
                          "value": str(inbound_comm_log_id), # Pass log ID
                         "action_id": edit_action_id
                     },
                      {
                         "type": "button",
                         "text": {
                             "type": "plain_text",
                             "text": "Reject",
                             "emoji": True
                         },
                         "style": "danger",
                          "value": str(inbound_comm_log_id), # Pass log ID
                         "action_id": reject_action_id
                     }
                 ]
            }
        ]
        
        return fallback_text, blocks 