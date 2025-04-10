import asyncio
import time
from loguru import logger
from googleapiclient.errors import HttpError

from app.config.settings import Settings
from anita.services.email_service import EmailService
from anita.services.inbound_email_service import InboundEmailService

# --- State Management (In-Memory - Lost on Restart) ---
# TODO: Persist this history ID in a database (Supabase) or file for robustness
last_history_id = None 

async def initialize_history_id(gmail_service, user_id='me'):
    """Gets the current history ID to start polling from."""
    global last_history_id
    if last_history_id:
        return last_history_id
    try:
        logger.info("Initializing Gmail history ID...")
        profile = gmail_service.users().getProfile(userId=user_id).execute()
        last_history_id = profile.get('historyId')
        logger.info(f"Initial history ID set to: {last_history_id}")
        return last_history_id
    except HttpError as error:
        logger.error(f"Failed to initialize history ID: {error}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error initializing history ID: {e}")
        return None

async def poll_gmail_for_updates(
    settings: Settings,
    email_service: EmailService,
    inbound_service: InboundEmailService,
    poll_interval_seconds: int = 300 # 5 minutes
):
    """Periodically polls Gmail for new messages using history records."""
    global last_history_id
    user_id = 'me' # Assuming we poll the authenticated user's inbox
    
    logger.info("Starting Gmail polling service...")
    
    # Wait a moment for other services to potentially start
    await asyncio.sleep(10)

    # Get an initial Gmail service instance
    gmail_service = email_service.get_gmail_service()
    if not gmail_service:
        logger.error("Failed to get Gmail service on startup. Polling cannot start.")
        return # Stop if we can't authenticate

    # Initialize the history ID
    current_history_id = await initialize_history_id(gmail_service, user_id)
    if not current_history_id:
         logger.error("Failed to get initial history ID. Polling cannot start reliably.")
         return

    logger.info(f"Polling Gmail for user '{user_id}' every {poll_interval_seconds} seconds, starting from history ID {current_history_id}.")

    while True:
        try:
            start_time = time.monotonic()
            logger.debug(f"Checking Gmail history starting from ID: {current_history_id}")
            
            # Ensure we have a valid service instance (handles potential token expiry/refresh)
            gmail_service = email_service.get_gmail_service()
            if not gmail_service:
                 logger.warning("Skipping poll cycle: Failed to get Gmail service.")
                 await asyncio.sleep(poll_interval_seconds) 
                 continue

            history_response = gmail_service.users().history().list(
                userId=user_id,
                startHistoryId=current_history_id,
                historyTypes=['messageAdded'] # Only interested in new messages
            ).execute()
            
            history_records = history_response.get('history', [])
            next_page_token = history_response.get('nextPageToken') # Handle pagination if needed (unlikely for 5 min interval)
            
            new_messages = []
            if history_records:
                for record in history_records:
                    messages_added = record.get('messagesAdded', [])
                    for added_msg in messages_added:
                        msg = added_msg.get('message')
                        if msg and 'INBOX' in msg.get('labelIds', []): # Process only if it landed in INBOX
                            new_messages.append(msg['id'])
            
            if new_messages:
                unique_message_ids = sorted(list(set(new_messages))) # Ensure uniqueness and process roughly in order
                logger.info(f"Found {len(unique_message_ids)} new message(s) in INBOX.")
                
                for message_id in unique_message_ids:
                    try:
                        logger.debug(f"Fetching raw content for message ID: {message_id}")
                        # Fetch raw email bytes
                        msg_data = gmail_service.users().messages().get(
                            userId=user_id,
                            id=message_id,
                            format='raw'
                        ).execute()
                        
                        raw_email_bytes = msg_data.get('raw')
                        if raw_email_bytes:
                             # Decode from base64url
                             import base64
                             decoded_bytes = base64.urlsafe_b64decode(raw_email_bytes)
                             logger.info(f"Processing inbound email for message ID: {message_id}")
                             # Use background task if processing is long, but careful with async context
                             # For simplicity now, await directly
                             await inbound_service.process_inbound_email(decoded_bytes)
                        else:
                            logger.warning(f"Could not retrieve raw content for message ID: {message_id}")
                            
                    except HttpError as fetch_error:
                         logger.error(f"Error fetching message ID {message_id}: {fetch_error}")
                    except Exception as process_error:
                         logger.exception(f"Error processing message ID {message_id}: {process_error}")
                         
            else:
                logger.debug("No new messages found in INBOX history.")

            # Update history ID for the next poll
            new_history_id = history_response.get('historyId')
            if new_history_id:
                current_history_id = new_history_id
                last_history_id = current_history_id # Update global state (in-memory)
                logger.debug(f"Updated history ID to: {current_history_id}")
            else:
                logger.warning("No new history ID returned from Gmail API. Will reuse previous ID.")

        except HttpError as error:
            # Handle specific errors, e.g., 401 Unauthorized might mean token needs refresh/re-auth
            logger.error(f"HTTP error during Gmail poll: {error}")
            if error.resp.status == 401:
                logger.error("Gmail API returned 401 Unauthorized. Check credentials/authentication.")
                # Optionally stop polling or implement backoff
            # Add more specific error handling as needed
            
        except Exception as e:
            logger.exception(f"Unexpected error during Gmail poll cycle: {e}")
            # Implement backoff logic here if needed to avoid spamming logs on persistent errors
        
        # Wait for the next interval
        elapsed_time = time.monotonic() - start_time
        wait_time = max(0, poll_interval_seconds - elapsed_time)
        logger.debug(f"Polling cycle finished in {elapsed_time:.2f}s. Waiting {wait_time:.2f}s...")
        await asyncio.sleep(wait_time) 