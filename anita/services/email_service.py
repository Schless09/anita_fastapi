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

# Configuration
SENDER_EMAIL = "anita@recruitcha.com" # Or load from config
TOKEN_PICKLE_PATH = 'token.pkl'
CREDENTIALS_JSON_PATH = 'credentials.json' # Needed for refresh
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gmail_service():
    """Authenticates and returns the Gmail API service client.
    
    Prioritizes credentials from environment variables for deployment environments (Vercel).
    Falls back to token.pkl for local development.
    """
    creds = None
    
    # --- Vercel/Deployment Environment Check ---
    # Check for environment variables typically set in Vercel
    # VERCEL_ENV is a common indicator, but you might use others
    is_deployed = os.environ.get('VERCEL') == '1' or os.environ.get('ENV') == 'production' # Add other indicators if needed
    
    if is_deployed:
        logger.info("Attempting to load Gmail credentials from environment variables for deployment.")
        refresh_token = os.environ.get('GMAIL_REFRESH_TOKEN')
        client_id = os.environ.get('GMAIL_CLIENT_ID')
        client_secret = os.environ.get('GMAIL_CLIENT_SECRET')
        token_uri = os.environ.get('GMAIL_TOKEN_URI', 'https://oauth2.googleapis.com/token') # Default token URI

        if refresh_token and client_id and client_secret:
            try:
                creds = Credentials(
                    None, # No access token initially, will be refreshed
                    refresh_token=refresh_token,
                    token_uri=token_uri,
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=SCOPES
                )
                # Check if refresh is needed (it likely is initially or if expired)
                if not creds.valid or creds.expired:
                     logger.info("Refreshing token using environment variables credentials.")
                     creds.refresh(Request())
                     logger.info("Token refreshed successfully (from env vars).")
            except Exception as e:
                logger.error(f"Failed to create/refresh credentials from environment variables: {e}")
                return None # Cannot proceed if env var loading fails in deployed env
        else:
            logger.error("Missing required Gmail credentials environment variables (GMAIL_REFRESH_TOKEN, GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET). Cannot authenticate in deployed environment.")
            return None
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
    if creds:
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
        logger.error("Failed to obtain valid credentials.")
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
    
    Args:
        recipient_email: The email address of the candidate.
        candidate_name: The name of the candidate (used for personalization).
        job_matches: A list of dictionaries, where each dict contains at least
                     'job_title' and 'job_url'.
        candidate_id: The UUID of the candidate.
        supabase_client: An initialized async Supabase client instance.
    """
    service = get_gmail_service()
    if not service:
        logger.error("Failed to get Gmail service. Cannot send email.")
        return False # Indicate failure

    if not job_matches:
        logger.warning(f"No job matches provided for {recipient_email}. Email not sent.")
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
        title = match.get('job_title', 'N/A')
        url = match.get('job_url', '#')
        plain_text_parts.append(f"- {title}: {url}")
        html_parts.append(f'<li><a href="{url}">{title}</a></li>')

    plain_text_parts.append("\nWe encourage you to check them out! Please reply to this email if you have any questions.")
    plain_text_parts.append("\n\nBest regards,\nAnita, Your personal career co-pilot")

    html_parts.append("</ul><p>We encourage you to check them out! Please reply to this email if you have any questions.</p>")
    html_parts.append("<p>Best regards,<br>Anita, Your personal career co-pilot</p></body></html>")

    plain_text = "\n".join(plain_text_parts)
    html_content = "\n".join(html_parts)

    message = create_message(SENDER_EMAIL, recipient_email, subject, plain_text, html_content)

    sent_message_details = send_message(service, 'me', message)
    if sent_message_details:
        logger.info(f"Successfully sent job match email to {recipient_email}")
        
        # Log the communication in the database
        try:
            communication_log = {
                "candidates_id": str(candidate_id),  # Convert UUID to string
                "type": "email",  # Must be one of: 'email', 'call', 'sms', 'iMessage'
                "direction": "outbound",  # Must be either 'inbound' or 'outbound'
                "subject": subject,
                "content": plain_text,  # Store the plain text version
                "metadata": {
                    "message_id": sent_message_details.get('id'),
                    "recipient": recipient_email,
                    "html_content": html_content,
                    "job_matches": job_matches
                }
                # timestamp will default to now() in DB
            }
            
            log_resp = await supabase_client.table("communications_dev").insert(communication_log).execute()
            if hasattr(log_resp, 'data') and log_resp.data:
                logger.info(f"Successfully logged email communication for candidate {candidate_id}")
            else:
                logger.warning(f"Could not log email communication for candidate {candidate_id}. Response: {log_resp}")
                
        except Exception as log_err:
            logger.error(f"Error logging email communication for candidate {candidate_id}: {log_err}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return True
    
    return False

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
        success = send_job_match_email(test_recipient, test_name, test_matches)
        if success:
            logger.info("--- Test Email Sent Successfully (check inbox) ---")
        else:
            logger.error("--- Test Email Failed to Send --- ")
    else:
        logger.error(f"'{TOKEN_PICKLE_PATH}' not found. Run auth.py to generate it before testing.")
        print(f"'{TOKEN_PICKLE_PATH}' not found. Run auth.py to generate it before testing.") 