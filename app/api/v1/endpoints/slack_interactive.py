import json
from fastapi import APIRouter, Depends, Request, HTTPException, Form, BackgroundTasks
from fastapi.responses import Response
from slack_sdk.signature import SignatureVerifier
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
from loguru import logger
import urllib.parse

from app.config.settings import Settings, get_settings
from anita.services.inbound_email_service import InboundEmailService
from app.config.supabase import get_supabase_client
from supabase import AsyncClient
from app.config.utils import get_table_name # Corrected import path

router = APIRouter()

# Action IDs (should match those in slack_service.py)
APPROVE_ACTION_ID = "approve_email_reply"
REJECT_ACTION_ID = "reject_email_reply"
EDIT_ACTION_ID = "edit_email_reply"
EDIT_MODAL_CALLBACK_ID = "edit_email_reply_modal"

# Helper function to verify Slack requests
async def verify_slack_request(request: Request, settings: Settings):
    verifier = SignatureVerifier(settings.slack_signing_secret)
    body = await request.body()
    headers = request.headers

    if not verifier.is_valid_request(body, headers):
        logger.warning("Invalid Slack signature received.")
        raise HTTPException(status_code=403, detail="Invalid Slack signature")
    return body # Return body for parsing later

# Dependency to get InboundEmailService (simplified)
# Assumes EmailService/SlackService can be instantiated within
async def get_inbound_service(settings: Settings = Depends(get_settings), supabase_client: AsyncClient = Depends(get_supabase_client)) -> InboundEmailService:
    from anita.services.email_service import EmailService
    from anita.services.slack_service import SlackService # Corrected import path
    email_service = EmailService(settings)
    slack_service = SlackService(settings)
    return InboundEmailService(settings, supabase_client, email_service, slack_service)

async def handle_approve_action(payload: dict, background_tasks: BackgroundTasks, inbound_service: InboundEmailService, settings: Settings):
    user_id = payload['user']['id']
    action = payload['actions'][0]
    inbound_log_id = int(action['value'])
    response_url = payload['response_url']
    
    logger.info(f"User {user_id} clicked Approve for log ID {inbound_log_id}")

    # 1. Fetch the proposed reply from the log
    communications_table = get_table_name("communications", settings)
    log_resp = await inbound_service.supabase.table(communications_table)\
                            .select("metadata")\
                            .eq("id", inbound_log_id)\
                            .maybe_single().execute()
    
    if not log_resp.data or not log_resp.data.get("metadata") or not log_resp.data["metadata"].get("proposed_ai_reply"):
        logger.error(f"Could not retrieve proposed reply for log ID {inbound_log_id} during approval.")
        # Notify user in Slack
        slack_client = WebClient(token=settings.slack_bot_token)
        slack_client.chat_postEphemeral(channel=payload['channel']['id'], user=user_id, text=f":warning: Error: Could not find the proposed reply data for Log ID {inbound_log_id}. Please check logs.")
        return

    proposed_reply = log_resp.data["metadata"]["proposed_ai_reply"]
    
    # 2. Add email sending to background tasks
    background_tasks.add_task(inbound_service.send_approved_email, inbound_log_id, proposed_reply, user_id)
    
    # 3. Update the original Slack message (optional but good UX)
    try:
        slack_client = WebClient(token=settings.slack_bot_token)
        original_blocks = payload['message']['blocks']
        # Remove the actions block and add an update message
        updated_blocks = [block for block in original_blocks if block.get('type') != 'actions']
        updated_blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":white_check_mark: Approved by <@{user_id}>. Sending email..."
                }
            ]
        })
        slack_client.chat_update(
            channel=payload['channel']['id'],
            ts=payload['message']['ts'],
            blocks=updated_blocks,
            text=f"Email reply approved by {user_id}"
        )
        logger.info(f"Updated Slack message for approved log ID {inbound_log_id}")
    except SlackApiError as e:
        logger.error(f"Error updating Slack message after approval: {e.response['error']}")
    except Exception as e:
         logger.exception(f"Unexpected error updating Slack message after approval: {e}")

async def handle_reject_action(payload: dict, background_tasks: BackgroundTasks, inbound_service: InboundEmailService, settings: Settings):
    user_id = payload['user']['id']
    action = payload['actions'][0]
    inbound_log_id = int(action['value'])
    
    logger.info(f"User {user_id} clicked Reject for log ID {inbound_log_id}")
    
    # 1. Update the communication log status
    background_tasks.add_task(inbound_service._update_log_metadata, inbound_log_id, {"status": "rejected_by_slack", "rejected_by": user_id})
    
    # 2. Update the original Slack message
    try:
        slack_client = WebClient(token=settings.slack_bot_token)
        original_blocks = payload['message']['blocks']
        # Remove the actions block and add an update message
        updated_blocks = [block for block in original_blocks if block.get('type') != 'actions']
        updated_blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":x: Rejected by <@{user_id}>. Email will not be sent."
                }
            ]
        })
        slack_client.chat_update(
            channel=payload['channel']['id'],
            ts=payload['message']['ts'],
            blocks=updated_blocks,
            text=f"Email reply rejected by {user_id}"
        )
        logger.info(f"Updated Slack message for rejected log ID {inbound_log_id}")
    except SlackApiError as e:
        logger.error(f"Error updating Slack message after rejection: {e.response['error']}")
    except Exception as e:
         logger.exception(f"Unexpected error updating Slack message after rejection: {e}")
         
async def handle_edit_action(payload: dict, settings: Settings):
    user_id = payload['user']['id']
    action = payload['actions'][0]
    inbound_log_id = int(action['value'])
    trigger_id = payload['trigger_id']
    
    logger.info(f"User {user_id} clicked Edit for log ID {inbound_log_id}")

    # 1. Fetch original proposed reply to pre-fill the modal
    communications_table = get_table_name("communications", settings)
    # Use synchronous client for simplicity within this function or pass async client
    supabase_sync = settings.get_supabase_client() # Assuming a sync client getter exists or create one
    log_resp = await get_supabase_client().table(communications_table)\
                            .select("metadata")\
                            .eq("id", inbound_log_id)\
                            .maybe_single().execute()
                            
    if not log_resp.data or not log_resp.data.get("metadata") or not log_resp.data["metadata"].get("proposed_ai_reply"):
        logger.error(f"Could not retrieve proposed reply for log ID {inbound_log_id} to open edit modal.")
        # Maybe post an ephemeral message back? 
        # For now, just open modal with empty text area if fetch fails.
        proposed_reply = "" 
    else:
        proposed_reply = log_resp.data["metadata"]["proposed_ai_reply"]

    # 2. Open a Slack modal
    try:
        slack_client = WebClient(token=settings.slack_bot_token)
        modal_view = {
            "type": "modal",
            "callback_id": EDIT_MODAL_CALLBACK_ID,
            "private_metadata": str(inbound_log_id), # Pass log ID securely
            "title": {"type": "plain_text", "text": "Edit Email Reply"},
            "submit": {"type": "plain_text", "text": "Send Edited Reply"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "edited_reply_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "edited_reply_input",
                        "multiline": True,
                        "initial_value": proposed_reply
                    },
                    "label": {"type": "plain_text", "text": "Email Content"}
                }
            ]
        }
        slack_client.views_open(trigger_id=trigger_id, view=modal_view)
        logger.info(f"Opened edit modal for log ID {inbound_log_id}")
    except SlackApiError as e:
        logger.error(f"Error opening edit modal: {e.response['error']}")
    except Exception as e:
         logger.exception(f"Unexpected error opening Slack modal: {e}")

async def handle_edit_modal_submission(payload: dict, background_tasks: BackgroundTasks, inbound_service: InboundEmailService, settings: Settings):
    user_id = payload['user']['id']
    inbound_log_id = int(payload['view']['private_metadata'])
    edited_reply = payload['view']['state']['values']['edited_reply_block']['edited_reply_input']['value']
    
    logger.info(f"User {user_id} submitted edited reply for log ID {inbound_log_id}")

    # Add email sending to background tasks with the *edited* content
    background_tasks.add_task(inbound_service.send_approved_email, inbound_log_id, edited_reply, user_id)
    
    # Optionally, update the original message or post a confirmation
    try:
        slack_client = WebClient(token=settings.slack_bot_token)
        # Find the original message? This is hard without storing message_ts.
        # Instead, post a new message confirming the action.
        slack_client.chat_postMessage(
            channel=settings.slack_reply_approval_channel_id, # Post back to the main channel
            text=f":pencil2: Edited reply for Log ID {inbound_log_id} submitted by <@{user_id}>. Sending email..."
        )
        logger.info(f"Posted confirmation for edited reply submission (Log ID: {inbound_log_id})")
    except SlackApiError as e:
         logger.error(f"Error posting modal submission confirmation: {e.response['error']}")
    except Exception as e:
         logger.exception(f"Unexpected error posting modal submission confirmation: {e}")

@router.post("/interactive")
async def handle_slack_interaction(
    request: Request,
    background_tasks: BackgroundTasks, 
    settings: Settings = Depends(get_settings),
    inbound_service: InboundEmailService = Depends(get_inbound_service)
):
    """Handle interactive components (buttons, modals) from Slack."""
    body_bytes = await verify_slack_request(request, settings)
    # Slack sends payload as form data urlencoded
    form_data = await request.form()
    payload_str = form_data.get('payload')
    if not payload_str:
        logger.error("Missing 'payload' in Slack interaction request.")
        raise HTTPException(status_code=400, detail="Missing payload")
        
    try:
        payload = json.loads(payload_str)
        interaction_type = payload.get('type')
        
        logger.debug(f"Received Slack interaction type: {interaction_type}")

        if interaction_type == 'block_actions':
            if not payload.get('actions'): 
                 return Response(status_code=200) # Ack request, no action needed
                 
            action_id = payload['actions'][0]['action_id']
            logger.info(f"Handling block action: {action_id}")

            if action_id == APPROVE_ACTION_ID:
                await handle_approve_action(payload, background_tasks, inbound_service, settings)
            elif action_id == REJECT_ACTION_ID:
                await handle_reject_action(payload, background_tasks, inbound_service, settings)
            elif action_id == EDIT_ACTION_ID:
                 await handle_edit_action(payload, settings)
            else:
                logger.warning(f"Unhandled block action ID: {action_id}")
                
        elif interaction_type == 'view_submission':
             callback_id = payload['view']['callback_id']
             logger.info(f"Handling view submission: {callback_id}")
             
             if callback_id == EDIT_MODAL_CALLBACK_ID:
                  await handle_edit_modal_submission(payload, background_tasks, inbound_service, settings)
             else:
                 logger.warning(f"Unhandled view submission callback ID: {callback_id}")
        else:
            logger.warning(f"Unhandled Slack interaction type: {interaction_type}")

        # Acknowledge the interaction immediately
        return Response(status_code=200)

    except json.JSONDecodeError:
        logger.error("Failed to decode Slack interaction payload.")
        raise HTTPException(status_code=400, detail="Invalid payload format")
    except Exception as e:
        logger.exception(f"Error processing Slack interaction: {e}")
        # Don't reveal internal errors to Slack
        return Response(status_code=500, content="Internal Server Error processing interaction") 