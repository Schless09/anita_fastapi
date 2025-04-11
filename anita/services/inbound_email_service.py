import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import base64

from openai import AsyncOpenAI
from supabase import AsyncClient, create_client
from loguru import logger
from email.message import Message as EmailMessage # Use EmailMessage to avoid name clash
from email.parser import BytesParser
from email.policy import default as default_policy
import html2text # For converting HTML email to text

from app.config.settings import Settings # Fixed import path
from anita.services.email_service import EmailService
from anita.services.slack_service import SlackService
from app.config.utils import get_table_name # Ensure correct import path from config/utils


class InboundEmailService:
    def __init__(self, settings: Settings, supabase_client: AsyncClient, email_service: EmailService, slack_service: SlackService):
        self.settings = settings
        self.supabase = supabase_client
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.email_service = email_service
        self.slack_service = slack_service
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False

    def _parse_email(self, raw_email: bytes) -> Optional[Dict[str, Any]]:
        """Parses raw email bytes into a structured dictionary."""
        try:
            parser = BytesParser(policy=default_policy)
            msg: EmailMessage = parser.parsebytes(raw_email)

            from_addr = msg.get('From')
            to_addr = msg.get('To')
            subject = msg.get('Subject', '[No Subject]')
            message_id = msg.get('Message-ID')
            in_reply_to = msg.get('In-Reply-To')
            references = msg.get('References')
            date_str = msg.get('Date')

            body_text = ""
            body_html = ""

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get('Content-Disposition'))

                    if "attachment" not in content_disposition:
                        if content_type == 'text/plain' and not body_text:
                            try:
                                body_text = part.get_payload(decode=True).decode(part.get_content_charset('utf-8'), errors='replace')
                            except (LookupError, TypeError):
                                body_text = part.get_payload(decode=True).decode('utf-8', errors='replace') # Fallback
                        elif content_type == 'text/html' and not body_html:
                            try:
                                body_html = part.get_payload(decode=True).decode(part.get_content_charset('utf-8'), errors='replace')
                            except (LookupError, TypeError):
                                body_html = part.get_payload(decode=True).decode('utf-8', errors='replace') # Fallback
            else:
                content_type = msg.get_content_type()
                if content_type == 'text/plain':
                    try:
                        body_text = msg.get_payload(decode=True).decode(msg.get_content_charset('utf-8'), errors='replace')
                    except (LookupError, TypeError):
                        body_text = msg.get_payload(decode=True).decode('utf-8', errors='replace')
                elif content_type == 'text/html':
                    try:
                        body_html = msg.get_payload(decode=True).decode(msg.get_content_charset('utf-8'), errors='replace')
                    except (LookupError, TypeError):
                        body_html = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            
            # Prefer text, but convert HTML if text is empty
            if not body_text and body_html:
                 body_text = self.html_converter.handle(body_html)
            
            # Basic cleaning: remove quoted replies (optional, refine as needed)
            # This is a simple heuristic, might need more robust library for complex cases
            cleaned_body = "\n".join(line for line in body_text.splitlines() if not line.strip().startswith('>'))
            
            # Extract email address from From field (handle names like "John Doe <john.doe@example.com>")
            match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', from_addr)
            sender_email = match.group(0) if match else from_addr # Fallback to full field
            
            return {
                "sender_email": sender_email.lower(),
                "recipient_email": to_addr, # Keep original for potential routing
                "subject": subject,
                "body_text": cleaned_body, # Use the cleaned body
                "body_html": body_html,
                "message_id": message_id,
                "in_reply_to": in_reply_to,
                "references": references,
                "date": date_str
            }
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return None

    async def _get_context_from_thread(self, parsed_email: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Finds the candidate and relevant job context based on email headers or sender."""
        sender_email = parsed_email["sender_email"]
        in_reply_to = parsed_email["in_reply_to"]
        references = parsed_email["references"]
        # Use a reference Message-ID if available (more reliable for threading)
        thread_ref_id = in_reply_to or (references.split()[0] if references else None)
        
        candidates_table = get_table_name("candidates", self.settings)
        communications_table = get_table_name("communications", self.settings)
        jobs_table = get_table_name("jobs", self.settings)
        
        candidate_id = None
        candidate_name = None
        communicated_job_ids = set()
        thread_id = None

        try:
            # 1. Try finding the communication thread
            if thread_ref_id:
                logger.info(f"Searching communication log for reference ID: {thread_ref_id}")
                log_resp = await self.supabase.table(communications_table)\
                                .select("id, candidates_id, thread_id, metadata")\
                                .eq("metadata->>message_id", thread_ref_id)\
                                .limit(1)\
                                .execute()
                if log_resp.data:
                    log_entry = log_resp.data[0]
                    candidate_id = log_entry['candidates_id']
                    thread_id = log_entry.get('thread_id') # Get the existing thread_id
                    # Extract job IDs from the specific message it replies to
                    if log_entry.get('metadata') and log_entry['metadata'].get('job_matches_sent'):
                         communicated_job_ids.update([j['job_id'] for j in log_entry['metadata']['job_matches_sent'] if j.get('job_id')])
                    logger.info(f"Found candidate {candidate_id} and thread {thread_id} via reference ID.")
                else:
                    logger.warning(f"No communication log found for reference ID: {thread_ref_id}. Will try sender email.")
            
            # 2. If no thread found via ID, try finding candidate by sender email
            if not candidate_id:
                logger.info(f"Searching candidate by email: {sender_email}")
                cand_resp = await self.supabase.table(candidates_table)\
                                .select("id, full_name")\
                                .eq("email", sender_email)\
                                .limit(1)\
                                .execute()
                if cand_resp.data:
                    candidate_info = cand_resp.data[0]
                    candidate_id = candidate_info['id']
                    candidate_name = candidate_info.get('full_name', 'Candidate')
                    logger.info(f"Found candidate {candidate_id} via email.")
                    # If found by email, we need to find *all* jobs communicated to them recently?
                    # This is less precise. Let's try finding the latest relevant thread instead.
                    # We might need a thread_id established earlier in the conversation.
                    # For now, let's try finding the most recent outbound email thread to this candidate.
                    latest_thread_resp = await self.supabase.table(communications_table)\
                        .select("thread_id, metadata")\
                        .eq("candidates_id", candidate_id)\
                        .eq("direction", "outbound")\
                        .limit(1)\
                        .execute()
                    if latest_thread_resp.data:
                         latest_log = latest_thread_resp.data[0]
                         thread_id = latest_log.get('thread_id')
                         if latest_log.get('metadata') and latest_log['metadata'].get('job_matches_sent'):
                             communicated_job_ids.update([j['job_id'] for j in latest_log['metadata']['job_matches_sent'] if j.get('job_id')])
                         logger.info(f"Inferred thread {thread_id} and jobs from latest outbound for candidate {candidate_id}")
                    else:
                        logger.warning(f"Could not find any recent outbound communication for candidate {candidate_id} to establish context.")
                        # No context can be reliably determined
                        return None
                else:
                    logger.error(f"Candidate not found for email: {sender_email}. Cannot process reply.")
                    return None # Cannot proceed without identified candidate

            # 3. Fetch Candidate Name if not already fetched
            if not candidate_name and candidate_id:
                 cand_resp = await self.supabase.table(candidates_table)\
                                .select("full_name")\
                                .eq("id", candidate_id)\
                                .limit(1).execute()
                 if cand_resp.data:
                     candidate_name = cand_resp.data[0].get('full_name', 'Candidate')

            # 4. Fetch details for the communicated jobs
            allowed_job_details = []
            if communicated_job_ids:
                logger.info(f"Fetching details for communicated job IDs: {communicated_job_ids}")
                
                # Select additional relevant job columns for AI context
                select_columns = (
                    "id, job_title, company_name, product_description, job_url, "
                    "key_responsibilities, seniority, work_arrangement, location_city, "
                    "location_state, tech_stack_must_haves, salary_range_min, salary_range_max, visa_sponsorship"
                )
                
                jobs_resp = await self.supabase.table(jobs_table)\
                                .select(select_columns)\
                                .in_("id", list(communicated_job_ids))\
                                .execute()
                if jobs_resp.data:
                    allowed_job_details = jobs_resp.data
                    logger.info(f"Fetched details for {len(allowed_job_details)} allowed jobs.")
                else:
                    logger.warning(f"Could not fetch details for communicated job IDs: {communicated_job_ids}")
            else:
                 logger.warning(f"No communicated job IDs found for candidate {candidate_id} in this context.")

            # Generate a new thread_id if none was found (e.g., first reply found via email only)
            if not thread_id:
                 thread_id = str(uuid.uuid4())
                 logger.info(f"Generated new thread_id for this interaction: {thread_id}")

            return {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name or "Candidate",
                "thread_id": thread_id,
                "allowed_jobs": allowed_job_details
            }

        except Exception as e:
            logger.exception(f"Error getting context for email from {sender_email}: {e}")
            return None

    async def _generate_ai_reply(self, candidate_message: str, context: Dict[str, Any]) -> Optional[str]:
        """Generates a reply using OpenAI, constrained by the provided context."""
        candidate_name = context["candidate_name"]
        allowed_jobs = context["allowed_jobs"]

        if not allowed_jobs:
            # If no specific jobs were discussed previously or found, provide a generic response
            prompt = f"""
You are Anita, a friendly AI career co-pilot.
A candidate named {candidate_name} has replied to one of your previous emails, but the context doesn't specify which job openings were discussed.
Their message is: "{candidate_message}"

Please respond politely. Acknowledge their message. Ask them to clarify which role they are asking about or what specific information they need. Reassure them you want to help once you have more details.
Do not mention any specific job titles or companies unless they mentioned them clearly first.
Keep the response concise and helpful.
"""
        else:
            job_details_parts = []
            for job in allowed_jobs:
                job_str = (
                    f"- Job Title: {job.get('job_title', 'N/A')}\\n"
                    f"  Company: {job.get('company_name', 'N/A')}\\n"
                    f"  Description Snippet: {job.get('product_description', 'N/A')[:200]}...\\n"
                    f"  URL: {job.get('job_url', 'N/A')}"
                )
                job_details_parts.append(job_str)
            job_context_str = "\\n".join(job_details_parts)

            prompt = f"""
You are Anita, a friendly and helpful AI career co-pilot.
You are replying to an email from a candidate named {candidate_name}.
Their message is: "{candidate_message}"

You have previously discussed the following job opportunity/opportunities with them. You can ONLY discuss these specific roles:
{job_context_str}

**Instructions:**
1. Carefully read the candidate's message.
2. Answer any questions they have ONLY about the jobs listed above. Use the provided details.
3. If they ask about jobs NOT listed above, politely state that you can only provide information on the roles previously shared and listed here.
4. If they ask general questions (e.g., about the process), answer helpfully but briefly.
5. Do NOT invent information or discuss internal processes not meant for candidates.
6. Do NOT mention other candidates.
7. Maintain a positive, professional, and encouraging tone.
8. Keep the reply concise and focused.
9. Sign off as "Anita, Your personal career co-pilot".

Generate the email reply body based *only* on the candidate's message and the allowed job information provided above.
"""

        try:
            logger.info("Generating AI reply...")
            response = await self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=self.settings.openai_temperature,
                max_tokens=500
            )
            reply_content = response.choices[0].message.content.strip()
            logger.info("AI reply generated successfully.")
            return reply_content
        except Exception as e:
            logger.exception(f"Error generating AI reply: {e}")
            return None

    async def _log_communication(self, context: Dict[str, Any], parsed_email: Dict[str, Any], ai_reply: Optional[str], direction: str):
        """Logs the email interaction to the database."""
        communications_table = get_table_name("communications", self.settings)
        log_entry = {
            "candidates_id": context.get("candidate_id"),
            "thread_id": context.get("thread_id"),
            "type": "email",
            "direction": direction,
            "subject": parsed_email.get("subject"),
            "content": parsed_email.get("body_text") if direction == "inbound" else ai_reply,
            "metadata": {
                "message_id": parsed_email.get("message_id") if direction == "inbound" else None, # Store inbound message ID
                "reply_message_id": None, # Will be set when reply is sent
                "sender": parsed_email.get("sender_email") if direction == "inbound" else self.settings.sender_email,
                "recipient": parsed_email.get("recipient_email") if direction == "inbound" else parsed_email.get("sender_email"),
                "ai_generated": direction == "outbound",
                "allowed_jobs_context": context.get("allowed_jobs") if direction == "outbound" else None, # Log context used for AI reply
                "in_reply_to": parsed_email.get("in_reply_to"),
                "references": parsed_email.get("references"),
            }
        }
        try:
            log_resp = await self.supabase.table(communications_table).insert(log_entry).execute()
            if hasattr(log_resp, 'data') and log_resp.data:
                logger.info(f"Successfully logged {direction} email communication for candidate {context.get('candidate_id')}, thread {context.get('thread_id')}")
                return log_resp.data[0].get('id') # Return log entry ID
            else:
                 logger.error(f"Error logging {direction} email communication: {log_resp.error or log_resp}")
                 return None
        except Exception as e:
            logger.exception(f"Exception logging {direction} email communication: {e}")
            return None

    async def _update_log_metadata(self, log_id: int, metadata_update: dict):
        """Merges new data into the metadata JSONB field of a specific communication log."""
        communications_table = get_table_name("communications", self.settings)
        try:
            # Fetch existing metadata
            resp = await self.supabase.table(communications_table).select("metadata").eq("id", log_id).maybe_single().execute()
            if not resp.data:
                 logger.error(f"Cannot update metadata: Log entry with ID {log_id} not found.")
                 return False
            
            existing_metadata = resp.data.get("metadata", {})
            if not isinstance(existing_metadata, dict):
                logger.warning(f"Existing metadata for log {log_id} is not a dict ({type(existing_metadata)}). Overwriting.")
                existing_metadata = {}
            
            # Merge new data
            existing_metadata.update(metadata_update)

            # Update the record
            update_resp = await self.supabase.table(communications_table)\
                                     .update({"metadata": existing_metadata})\
                                     .eq("id", log_id)\
                                     .execute()

            if update_resp.data:
                logger.info(f"Successfully updated metadata for communication log {log_id}")
                return True
            else:
                logger.error(f"Failed to update metadata for communication log {log_id}: {update_resp.error}")
                return False
        except Exception as e:
            logger.exception(f"Exception updating metadata for communication log {log_id}: {e}")
            return False

    async def process_inbound_email(self, raw_email_bytes: bytes):
        """Main function to process a new inbound email."""
        logger.info("--- Processing new inbound email --- ")

        # 1. Parse the email
        parsed_email = self._parse_email(raw_email_bytes)
        if not parsed_email:
            logger.error("Failed to parse inbound email. Aborting.")
            return
        logger.info(f"Parsed email from: {parsed_email['sender_email']}, Subject: {parsed_email['subject']}")

        # 2. Get context (candidate, allowed jobs)
        context = await self._get_context_from_thread(parsed_email)
        if not context or not context.get("candidate_id"):
            logger.warning(f"Could not determine context or candidate for email from {parsed_email['sender_email']}. Skipping AI reply.")
            # Optionally log the raw inbound email even if context fails
            # await self._log_communication({"thread_id": str(uuid.uuid4())}, parsed_email, None, "inbound")
            return
        logger.info(f"Context found: Candidate ID {context['candidate_id']}, Thread ID {context['thread_id']}, Allowed Jobs: {len(context['allowed_jobs'])}")

        # 3. Log the inbound email (initially without AI reply)
        inbound_log_id = await self._log_communication(context, parsed_email, None, "inbound")
        if not inbound_log_id:
             logger.error("Failed to log inbound email. Aborting processing.")
             return # Cannot proceed without a log ID to reference

        # 4. Generate AI reply
        ai_reply = await self._generate_ai_reply(parsed_email["body_text"], context)
        if not ai_reply:
            logger.error("Failed to generate AI reply. No reply will be sent or notified.")
            # Optionally update log status here to indicate AI failure
            await self._update_log_metadata(inbound_log_id, {"status": "ai_reply_failed"})
            return
        logger.info("AI reply generated.")
        
        # 4.5 Store proposed AI reply in the *inbound* log's metadata for later retrieval
        metadata_update = {
            "proposed_ai_reply": ai_reply,
            "status": "pending_approval" # Add status
        }
        await self._update_log_metadata(inbound_log_id, metadata_update)
        
        # 5. Send Slack Notification for Approval/Action (Only in Staging/Production)
        if self.settings.environment in ["staging", "production"]:
            logger.info(f"Environment is '{self.settings.environment}', proceeding with Slack notification.")
            try:
                slack_channel_id = self.settings.slack_reply_approval_channel_id
                if slack_channel_id and self.slack_service:
                    fallback_text, blocks = self.slack_service.format_reply_notification(
                        candidate_email=parsed_email["sender_email"],
                        candidate_message=parsed_email["body_text"],
                        ai_reply=ai_reply,
                        inbound_comm_log_id=inbound_log_id # Pass the log ID
                    )
                    logger.info(f"Sending Slack notification to channel {slack_channel_id} for log ID {inbound_log_id}")
                    self.slack_service.send_notification(
                       channel_id=slack_channel_id, 
                       text=fallback_text, 
                       blocks=blocks
                    )
                    # Update log status
                    await self._update_log_metadata(inbound_log_id, {"status": "pending_slack_action"})
                else:
                     logger.warning("Slack channel ID not configured or Slack service unavailable. Skipping notification.")
                     # Update log status
                     await self._update_log_metadata(inbound_log_id, {"status": "slack_config_missing"})
            except Exception as e:
                 logger.exception(f"Error sending Slack notification: {e}") 
                 # Update log status
                 await self._update_log_metadata(inbound_log_id, {"status": "slack_notification_failed"})
        else:
            logger.info(f"Environment is '{self.settings.environment}', skipping Slack notification for email reply approval.")
            # Optional: Update status to indicate automatic skip if needed
            # await self._update_log_metadata(inbound_log_id, {"status": "slack_skipped_dev"})

        # --- REMOVED STEP 6: Automatic Email Sending --- 
        # Email sending is now triggered by Slack actions handled by a separate endpoint.

        logger.info(f"--- Finished initial processing for inbound email (Log ID: {inbound_log_id}). Waiting for Slack action. ---")

    async def send_approved_email(self, inbound_log_id: int, final_reply_content: str, approver_user_id: Optional[str] = None):
        """Sends the email reply after approval/edit via Slack."""
        logger.info(f"Attempting to send approved email for inbound log ID: {inbound_log_id}")
        communications_table = get_table_name("communications", self.settings)
        
        try:
            # 1. Fetch the original inbound log entry to get context
            log_resp = await self.supabase.table(communications_table)\
                                    .select("id, candidates_id, thread_id, subject, metadata")\
                                    .eq("id", inbound_log_id)\
                                    .maybe_single()\
                                    .execute()
            
            if not log_resp.data:
                logger.error(f"Could not find communication log entry with ID: {inbound_log_id}. Cannot send reply.")
                # TODO: Notify Slack channel about this failure?
                return False
            
            inbound_log = log_resp.data
            metadata = inbound_log.get("metadata", {})
            if not isinstance(metadata, dict):
                 logger.error(f"Invalid metadata format in log {inbound_log_id}. Cannot proceed.")
                 return False
            
            candidate_id = inbound_log.get("candidates_id")
            thread_id = inbound_log.get("thread_id")
            original_subject = inbound_log.get("subject", "")
            recipient_email = metadata.get("sender") # The sender of the inbound email is the recipient of the reply
            references = metadata.get("references")
            inbound_message_id = metadata.get("message_id") # ID of the email we are replying to

            if not recipient_email or not inbound_message_id:
                 logger.error(f"Missing recipient email or original message_id in log {inbound_log_id} metadata. Cannot send reply.")
                 return False

            # 2. Prepare and send the reply email
            reply_subject = f"Re: {original_subject}" if not original_subject.lower().startswith("re:") else original_subject
            
            # Use the EmailService's capability to handle threading headers
            sent_message_details = await self.email_service.send_reply_email(
                recipient_email=recipient_email,
                subject=reply_subject,
                plain_text_body=final_reply_content,
                # html_body=None, # Optional: generate HTML version if needed
                thread_references=references,
                thread_in_reply_to=inbound_message_id # Reply to the ID of the mail received
            )
            
            if sent_message_details and sent_message_details.get('id'):
                logger.info(f"Successfully sent approved reply email for log {inbound_log_id}. Reply Message ID: {sent_message_details.get('id')}")
                
                # 3. Log the final outbound communication
                outbound_log_entry_data = {
                    "candidates_id": candidate_id,
                    "thread_id": thread_id,
                    "type": "email",
                    "direction": "outbound",
                    "subject": reply_subject,
                    "content": final_reply_content,
                    "metadata": {
                        "reply_message_id": sent_message_details.get('id'),
                        "sender": self.settings.sender_email,
                        "recipient": recipient_email,
                        "ai_generated": metadata.get("proposed_ai_reply") == final_reply_content, # Track if it was modified
                        "approved_by": approver_user_id, # Log who approved/sent it
                        "in_reply_to_log_id": inbound_log_id, # Link back to the inbound log
                        "in_reply_to_message_id": inbound_message_id,
                        "references": references,
                        "status": "sent_after_approval"
                    }
                }
                log_resp = await self.supabase.table(communications_table).insert(outbound_log_entry_data).execute()
                if log_resp.data:
                     logger.info(f"Successfully logged final outbound email communication for thread {thread_id}, linked to inbound {inbound_log_id}")
                else:
                     logger.error(f"Error logging final outbound email for thread {thread_id}: {log_resp.error}")

                # 4. Update the original inbound log status
                await self._update_log_metadata(inbound_log_id, {"status": "replied", "outbound_log_id": log_resp.data[0]['id'] if log_resp.data else None})
                return True
            else:
                logger.error(f"Failed to send approved reply email for log {inbound_log_id}.")
                # Update status to reflect send failure
                await self._update_log_metadata(inbound_log_id, {"status": "approved_send_failed"})
                return False
        except Exception as e:
            logger.exception(f"Error sending approved email for log {inbound_log_id}: {e}")
            # Attempt to update status
            await self._update_log_metadata(inbound_log_id, {"status": "approved_send_exception"})
            return False


# Helper function assuming you have a way to get dependencies
def get_inbound_email_service(settings: Settings, supabase_client: AsyncClient) -> InboundEmailService:
     # Need instances of EmailService and SlackService
     # This might be handled by your dependency injection framework (e.g., FastAPI Depends)
     email_service = EmailService(settings)
     slack_service = SlackService(settings)
     return InboundEmailService(settings, supabase_client, email_service, slack_service) 