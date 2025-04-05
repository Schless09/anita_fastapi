import pickle
import base64
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import os
import uuid
from supabase._async.client import AsyncClient
import traceback
from app.config.settings import get_settings # Keep this import
from app.config.utils import get_table_name # Import get_table_name from utils
import io # Add io import

# Configuration
SENDER_EMAIL = os.environ.get('GMAIL_USER', "anita@recruitcha.com")  # Use GMAIL_USER as sender email
TOKEN_PICKLE_PATH = 'token.pkl'
CREDENTIALS_JSON_PATH = 'credentials.json' # Needed for refresh
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Remove basicConfig - logging should be configured centrally
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gmail_service():
    """Authenticates and returns the Gmail API service client.
    
    Prioritizes credentials from the GMAIL_TOKEN_B64 environment variable for deployment.
    Falls back to token.pkl for local development.
    """
    creds = None
    
    # --- Vercel/Deployment Environment Check ---
    # Check for environment variables typically set in Vercel
    # Updated check for Render.com deployment
    is_deployed = os.environ.get('RENDER') == 'true' or os.environ.get('ENVIRONMENT') == 'production' # Use RENDER or ENVIRONMENT
    
    if is_deployed:
        logger.info("Running in deployed environment. Attempting to load Gmail credentials from GMAIL_TOKEN_B64 env var.")
        token_b64 = os.environ.get("GMAIL_TOKEN_B64")
        if not token_b64:
            logger.error("GMAIL_TOKEN_B64 environment variable not found. Cannot authenticate Gmail in deployed environment.")
            return None
        
        try:
            token_bytes = base64.b64decode(token_b64)
            # Use io.BytesIO to treat bytes as a file-like object for pickle
            creds = pickle.load(io.BytesIO(token_bytes))
            logger.info("Successfully loaded credentials from GMAIL_TOKEN_B64.")
        except (pickle.UnpicklingError, base64.binascii.Error, TypeError) as e:
            logger.error(f"Failed to decode or unpickle token from GMAIL_TOKEN_B64: {e}")
            return None
            
        # Check if the loaded token needs refreshing (might still be needed)
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing token loaded from GMAIL_TOKEN_B64.")
            try:
                # We still need client_id/secret for refresh, even when loading token
                client_id = os.environ.get('GMAIL_CLIENT_ID')
                client_secret = os.environ.get('GMAIL_CLIENT_SECRET')
                # token_uri = os.environ.get('GMAIL_TOKEN_URI', 'https://oauth2.googleapis.com/token') # Not needed directly
                
                if not client_id or not client_secret:
                     logger.error("Missing GMAIL_CLIENT_ID or GMAIL_CLIENT_SECRET env vars needed for token refresh.")
                     return None # Cannot refresh without these
                
                # Manually set the client info on the credentials object before refresh
                # --- REMOVE THESE LINES --- 
                # creds.token_uri = token_uri # Causes AttributeError: property 'token_uri' of 'Credentials' object has no setter
                # creds.client_id = client_id
                # creds.client_secret = client_secret
                # --- END REMOVAL --- 
                
                # Attempt refresh - it should use internal state or find env vars
                creds.refresh(Request())
                logger.info("Token refreshed successfully (originally from GMAIL_TOKEN_B64).")
                # Note: We cannot easily save the refreshed token back to the env var here.
                # The refreshed token will only live for the duration of this instance.
            except Exception as e:
                logger.error(f"Error refreshing token originally from GMAIL_TOKEN_B64: {e}")
                # If refresh fails, the existing creds might still work if not expired
                # Or if they were just loaded and haven't been used yet.
                # Proceed cautiously. If creds are truly invalid now, service build will fail.
        
    else:
        # --- Local Development Fallback (using token.pkl) ---
        logger.info("Running in local environment. Attempting to load Gmail credentials from token.pkl.")
        if os.path.exists(TOKEN_PICKLE_PATH):
            with open(TOKEN_PICKLE_PATH, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in.
        # Or refresh the token if it's expired.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired token from token.pkl.")
                    creds.refresh(Request())
                    logger.info("Token refreshed successfully (from token.pkl).")
                    # Save the refreshed credentials back to token.pkl
                    with open(TOKEN_PICKLE_PATH, 'wb') as token:
                        pickle.dump(creds, token)
                        logger.info(f"Saved refreshed token to {TOKEN_PICKLE_PATH}")
                except Exception as e:
                    logger.error(f"Error refreshing token from token.pkl: {e}. Run auth.py to re-authenticate.")
                    return None # Indicate failure
            else:
                # If token.pkl doesn't exist or is invalid without a refresh token
                logger.error(f"Credentials not found or invalid in {TOKEN_PICKLE_PATH}. Run auth.py first.")
                return None # Indicate failure

    # --- Build and Return Service --- 
    if creds and creds.valid: # Check creds are valid before building service
        try:
            service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail service built successfully.")
            return service
        except HttpError as error:
            logger.error(f'An error occurred building the service: {error}')
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred building the service: {e}")
            return None
    else:
        # This case should ideally be caught earlier, but serves as a final check
        logger.error("Failed to obtain valid credentials after attempting load/refresh.")
        return None


def create_message(sender, to, subject, message_text_plain, message_text_html):
    """Create a MIME message."""
    message = MIMEMultipart('alternative')
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    # Attach plain text part first
    message.attach(MIMEText(message_text_plain, 'plain'))
    # Attach HTML part
    message.attach(MIMEText(message_text_html, 'html'))

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw_message}

def send_message(service, user_id, message):
    """Send an email message."""
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        logger.info(f"Email sent successfully. Message ID: {sent_message['id']}")
        return sent_message
    except HttpError as error:
        logger.error(f'An error occurred sending email: {error}')
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred sending email: {e}")
        return None

async def send_job_match_email(
    recipient_email: str, 
    candidate_name: str | None, 
    job_matches: list[dict],
    candidate_id: uuid.UUID,
    supabase_client: AsyncClient
):
    """
    Sends an email to a candidate with their top job matches and logs the communication.
    """
    logger.info(f"[Email Service] Entered send_job_match_email for candidate {candidate_id}, recipient {recipient_email}")
    
    logger.info("[Email Service] Attempting to get Gmail service...")
    service = get_gmail_service()
    if not service:
        # Error is already logged within get_gmail_service
        logger.error("[Email Service] Failed to get Gmail service. Aborting email send.")
        return False # Indicate failure
    logger.info("[Email Service] Successfully obtained Gmail service object.")

    if not job_matches:
        logger.warning(f"[Email Service] No job matches provided for {recipient_email}. Email not sent.")
        return False

    subject = "🔥 Top matches for you!"
    
    # Use only the first name in the greeting
    first_name = candidate_name.split(' ')[0] if candidate_name else None
    greeting = f"Hi {first_name}," if first_name else "Hi there,"

    # Construct email body
    plain_text_parts = [
        f"{greeting}\n",
        "Great news! We found some roles that look like a strong fit for your profile:\n"
    ]
    html_parts = [
        f"<html><body><h2>{greeting}</h2>",
        "<p>Great news! We found some roles that look like a strong fit for your profile:</p><ul>"
    ]

    for match in job_matches:
        title = match.get('job_title', 'N/A') # Use job_title from the match dict passed by BrainAgent
        url = match.get('job_url', '#') # Use job_url from the match dict
        plain_text_parts.append(f"- {title}: {url}")
        html_parts.append(f'<li><a href="{url}">{title}</a></li>')

    plain_text_parts.append("\nWe encourage you to check them out! Please reply to this email if you have any questions.")
    plain_text_parts.append("\n\nBest regards,\nAnita, Your personal career co-pilot")

    html_parts.append("</ul><p>We encourage you to check them out! Please reply to this email if you have any questions.</p>")
    html_parts.append("<p>Best regards,<br>Anita, Your personal career co-pilot</p></body></html>")

    plain_text = "\n".join(plain_text_parts)
    html_content = "\n".join(html_parts)

    message = create_message(SENDER_EMAIL, recipient_email, subject, plain_text, html_content)

    logger.info(f"[Email Service] Attempting to send message via send_message helper for {recipient_email}...")
    sent_message_details = send_message(service, 'me', message)
    
    if sent_message_details:
        logger.info(f"[Email Service] send_message returned success for {recipient_email}. Message ID: {sent_message_details.get('id')}")
        
        # Log the communication in the database
        logger.info(f"[Email Service] Attempting to log communication to database for candidate {candidate_id}...")
        try:
            # Generate a new thread_id for this email communication
            new_thread_id = str(uuid.uuid4())
            # logger.info(f"Generated new thread_id for job match email: {new_thread_id}") # Less verbose log

            communication_log = {
                "candidates_id": str(candidate_id),  # Convert UUID to string
                "thread_id": new_thread_id,  # Add the generated thread_id
                "type": "email",  
                "direction": "outbound", 
                "subject": subject,
                "content": plain_text,  # Store the plain text version
                "metadata": {
                    "message_id": sent_message_details.get('id'),
                    "recipient": recipient_email,
                    "html_content": html_content,
                    # Log simplified job matches, avoid large objects if job_matches contains full details
                    "job_matches_sent": [{'job_id': m.get('job_id'), 'title': m.get('job_title')} for m in job_matches]
                }
            }
            
            table_name = get_table_name("communications")
            log_resp = await supabase_client.table(table_name).insert(communication_log).execute()
            
            # Check response status more reliably
            if hasattr(log_resp, 'data') and log_resp.data:
                logger.info(f"[Email Service] Successfully logged email communication for candidate {candidate_id}")
            elif hasattr(log_resp, 'error') and log_resp.error:
                 logger.error(f"[Email Service] Error logging email communication for candidate {candidate_id}. Supabase error: {log_resp.error}")
            else:
                logger.warning(f"[Email Service] Could not log email communication for candidate {candidate_id}. Response: {log_resp}")
                
        except Exception as log_err:
            logger.error(f"[Email Service] Exception logging email communication for candidate {candidate_id}: {log_err}")
            # Optionally log traceback: logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't return False here, email was sent, just logging failed.
        
        return True # Email sent successfully (logging failure is separate)
    else:
        # Error is already logged within send_message
        logger.error(f"[Email Service] send_message returned failure for {recipient_email}. Email not logged.")
        return False # Indicate email send failure

async def send_missed_call_email(
    recipient_email: str, 
    candidate_name: str | None, 
    candidate_id: uuid.UUID,
    supabase_client: AsyncClient
):
    """Sends a 'Sorry I missed you' email if the call dropped early."""
    settings = get_settings() # Get settings to access base URL
    service = get_gmail_service()
    if not service:
        logger.error("Failed to get Gmail service. Cannot send missed call email.")
        return False

    # Construct the callback URL
    # Ensure candidate_id is string for URL
    callback_url = f"{settings.webhook_base_url}/candidates/{str(candidate_id)}/request-callback"
    logger.info(f"Generated callback URL for {candidate_id}: {callback_url}")

    subject = "Sorry I missed you!"
    first_name = candidate_name.split(' ')[0] if candidate_name else None
    greeting = f"Hi {first_name}," if first_name else "Hi there,"

    plain_text = f"""
{greeting}

I tried calling but it seems I missed you.

No worries! Reply to this email to let me know a better time to reach you.

Best regards,
Anita, your personal career co-pilot
""".strip()

    html_content = f"""
<html><body>
<p>{greeting}</p>
<p>I tried calling but it seems I missed you.</p>
<p>No worries! Reply to this email to let me know a better time to reach you.</p>
<p>Best regards,<br>Anita, your personal career co-pilot</p>
</body></html>
""".strip()

    message = create_message(SENDER_EMAIL, recipient_email, subject, plain_text, html_content)
    sent_message_details = send_message(service, 'me', message)

    if sent_message_details:
        logger.info(f"Successfully sent missed call email to {recipient_email}")
        # Log communication
        try:
            new_thread_id = str(uuid.uuid4())
            communication_log = {
                "candidates_id": str(candidate_id),
                "thread_id": new_thread_id,
                "type": "email",
                "direction": "outbound",
                "subject": subject,
                "content": plain_text,
                "metadata": {"message_id": sent_message_details.get('id'), "recipient": recipient_email}
            }
            # Use get_table_name
            table_name = get_table_name("communications")
            log_resp = await supabase_client.table(table_name).insert(communication_log).execute()
            if hasattr(log_resp, 'data') and log_resp.data:
                logger.info(f"Successfully logged missed call email for {candidate_id}")
            else:
                logger.warning(f"Could not log missed call email for {candidate_id}. Response: {log_resp}")
        except Exception as log_err:
            logger.error(f"Error logging missed call email for {candidate_id}: {log_err}")
        return True
    return False

async def send_no_matches_email(
    recipient_email: str, 
    candidate_name: str | None, 
    candidate_id: uuid.UUID,
    supabase_client: AsyncClient
):
    """Sends an email when no suitable job matches are found after a call."""
    service = get_gmail_service()
    if not service:
        logger.error("Failed to get Gmail service. Cannot send no matches email.")
        return False

    subject = "Thanks for speaking with me"
    first_name = candidate_name.split(' ')[0] if candidate_name else None
    greeting = f"Hi {first_name}," if first_name else "Hi there,"

    plain_text = f"""
{greeting}

Thanks for taking the time to speak with me.

We've reviewed your profile against our current openings. While we don't have an immediate match that meets your preferences right now, we're constantly getting new roles.

We'll keep your profile active and reach out as soon as a suitable opportunity comes up!

Best regards,
Anita, your personal career co-pilot
""".strip()

    html_content = f"""
<html><body>
<p>{greeting}</p>
<p>Thanks for taking the time to speak with me.</p>
<p>We've reviewed your profile against our current openings. While we don't have an immediate match that meets your preferences right now, we're constantly getting new roles.</p>
<p>We'll keep your profile active and reach out as soon as a suitable opportunity comes up!</p>
<p>Best regards,<br>Anita, your personal career co-pilot</p>
</body></html>
""".strip()

    message = create_message(SENDER_EMAIL, recipient_email, subject, plain_text, html_content)
    sent_message_details = send_message(service, 'me', message)

    if sent_message_details:
        logger.info(f"Successfully sent no matches email to {recipient_email}")
        # Log communication
        try:
            new_thread_id = str(uuid.uuid4())
            communication_log = {
                "candidates_id": str(candidate_id),
                "thread_id": new_thread_id,
                "type": "email",
                "direction": "outbound",
                "subject": subject,
                "content": plain_text,
                "metadata": {"message_id": sent_message_details.get('id'), "recipient": recipient_email}
            }
            # Use get_table_name
            table_name = get_table_name("communications")
            log_resp = await supabase_client.table(table_name).insert(communication_log).execute()
            if hasattr(log_resp, 'data') and log_resp.data:
                logger.info(f"Successfully logged no matches email for {candidate_id}")
            else:
                logger.warning(f"Could not log no matches email for {candidate_id}. Response: {log_resp}")
        except Exception as log_err:
            logger.error(f"Error logging no matches email for {candidate_id}: {log_err}")
        return True
    return False

# --- NEW FUNCTION FOR CALL TOO SHORT --- 
async def send_call_too_short_email(
    recipient_email: str,
    candidate_name: str | None,
    candidate_id: uuid.UUID,
    supabase_client: AsyncClient
) -> bool:
    """
    Sends an email when the initial call was too short to gather sufficient info.
    """
    logger.info(f"[Email Service] Entered send_call_too_short_email for candidate {candidate_id}, recipient {recipient_email}")

    service = get_gmail_service()
    if not service:
        logger.error("[Email Service] Failed to get Gmail service. Aborting email send for call_too_short.")
        return False

    subject = "Following up on our chat"

    first_name = candidate_name.split(' ')[0] if candidate_name else None
    greeting = f"Hi {first_name}," if first_name else "Hi there,"

    plain_text_body = (
        f"{greeting}\n\n"
        "It was great speaking with you, but we\'ll need more information to complete your profile and match you with jobs. \n"
        "Let us know when\'s a good time to call you back.\n\n"
        "Best regards,\nAnita, Your personal career co-pilot"
    )

    html_body = (
        f"<html><body><h2>{greeting}</h2>"
        "<p>It was great speaking with you, but we\'ll need more information to complete your profile and match you with jobs.</p>"
        "<p>Let us know when\'s a good time to call you back.</p>"
        "<p>Best regards,<br>Anita, Your personal career co-pilot</p></body></html>"
    )

    message = create_message(SENDER_EMAIL, recipient_email, subject, plain_text_body, html_body)

    logger.info(f"[Email Service] Attempting to send call_too_short email to {recipient_email}...")
    sent_message_details = send_message(service, 'me', message)

    if sent_message_details:
        logger.info(f"[Email Service] send_message returned success for call_too_short email to {recipient_email}. Message ID: {sent_message_details.get('id')}")

        # Log the communication
        logger.info(f"[Email Service] Attempting to log call_too_short communication to database for candidate {candidate_id}...")
        try:
            new_thread_id = str(uuid.uuid4())
            communication_log = {
                "candidates_id": str(candidate_id),
                "thread_id": new_thread_id,
                "type": "email",
                "direction": "outbound",
                "subject": subject,
                "content": plain_text_body,
                "metadata": {
                    "message_id": sent_message_details.get('id'),
                    "recipient": recipient_email,
                    "html_content": html_body,
                    "reason": "call_too_short" # Add reason for this specific email
                }
            }

            table_name = get_table_name("communications")
            log_resp = await supabase_client.table(table_name).insert(communication_log).execute()

            if hasattr(log_resp, 'data') and log_resp.data:
                logger.info(f"[Email Service] Successfully logged call_too_short email communication for candidate {candidate_id}")
                return True # Email sent and logged successfully
            elif hasattr(log_resp, 'error') and log_resp.error:
                logger.error(f"[Email Service] Error logging call_too_short email communication for candidate {candidate_id}. Supabase error: {log_resp.error}")
                return False # Email sent, but logging failed
            else:
                logger.warning(f"[Email Service] Communication log response for call_too_short email did not contain expected data or error for candidate {candidate_id}. Response: {log_resp}")
                return False # Email sent, but logging status unclear
        except Exception as log_err:
            logger.error(f"[Email Service] Exception logging call_too_short email communication for candidate {candidate_id}: {log_err}\n{traceback.format_exc()}")
            return False # Email sent, but logging failed due to exception
    else:
        logger.error(f"[Email Service] send_message failed for call_too_short email to {recipient_email}. Email not sent.")
        return False # Email failed to send

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Ensure token.pkl exists by running auth.py first
    test_recipient = "harrison.franke@gmail.com" # Set to your test email
    test_name = "Test Candidate" # You can change this if you like
    test_matches = [
        {'job_title': 'Software Engineer, Backend (Test)', 'job_url': 'https://example.com/job1'},
        {'job_title': 'Data Scientist (Test)', 'job_url': 'https://example.com/job2'}
    ]
    
    logger.info(f"--- Running Email Service Test --- ")
    logger.info(f"Attempting to send test email to: {test_recipient}")
    
    if os.path.exists(TOKEN_PICKLE_PATH):
        # Call the function directly for testing
        # success = send_job_match_email(test_recipient, test_name, test_matches)
        # success = send_missed_call_email(test_recipient, test_name, uuid.uuid4(), None) # Requires Supabase client mock or connection
        # success = send_no_matches_email(test_recipient, test_name, uuid.uuid4(), None) # Requires Supabase client mock or connection
        # success = send_call_too_short_email(test_recipient, test_name, uuid.uuid4(), None) # Requires Supabase client mock or connection
        # if success:
        #     logger.info("--- Test Email Sent Successfully (check inbox) ---")
        # else:
        #     logger.error("--- Test Email Failed to Send --- ")
        pass # Avoid running actual send without proper setup/mocking
    else:
        logger.error(f"'{TOKEN_PICKLE_PATH}' not found. Run auth.py to generate it before testing.")
        print(f"'{TOKEN_PICKLE_PATH}' not found. Run auth.py to generate it before testing.") 