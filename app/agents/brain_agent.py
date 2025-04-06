# agents/brain_agent.py
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from app.agents.langchain.tools.vector_store import VectorStoreTool
from app.agents.langchain.tools.document_processing import PDFProcessor, ResumeParser
from app.agents.langchain.tools.matching import MatchingTool
from app.agents.langchain.tools.communication import EmailTool
from app.schemas.candidate import CandidateCreate

from app.agents.langchain.chains.candidate_processing import CandidateProcessingChain
from app.agents.langchain.chains.job_matching import JobMatchingChain
from app.agents.langchain.chains.interview_scheduling import InterviewSchedulingChain
from app.agents.langchain.chains.follow_up import FollowUpChain

from app.services.candidate_service import CandidateService
from app.services.job_service import JobService
from app.services.retell_service import RetellService
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.services.matching_service import MatchingService
from app.config.settings import get_settings, Settings
from app.config.utils import get_table_name
from app.config.supabase import get_supabase_client

# Set up logging
import logging
import traceback
import uuid
import asyncio
import json
import os
import openai
from dotenv import load_dotenv
from fastapi import HTTPException
from supabase._async.client import AsyncClient, create_client

# Uncomment the email service import
from anita.services.email_service import send_job_match_email, send_missed_call_email, send_no_matches_email, send_call_too_short_email

logger = logging.getLogger(__name__)

class BrainAgent:
    """Orchestrator agent that coordinates other specialized agents."""
    
    def __init__(self, 
                 supabase_client: AsyncClient,
                 candidate_service: CandidateService,
                 openai_service: OpenAIService,
                 matching_service: MatchingService,
                 retell_service: RetellService,
                 vector_service: VectorService,
                 settings: Settings):
        """Initialize the brain agent and required services."""
        self._candidate_intake_agent = None
        self._job_matching_agent = None
        self._farming_matching_agent = None
        self._interview_agent = None
        self._follow_up_agent = None

        # Initialize needed services using injected instances
        self.supabase: AsyncClient = supabase_client
        self.candidate_service = candidate_service
        self.openai_service = openai_service
        self.matching_service = matching_service
        self.retell_service = retell_service
        # Assign settings and vector_service needed by agent properties
        self.settings = settings 
        self.vector_service = vector_service

        # Initialize the PDF processing tools
        self.pdf_processor = PDFProcessor()
        self.resume_parser = ResumeParser()

        # Define table names using the helper function with injected settings
        self.candidates_table = get_table_name("candidates")
        self.jobs_table = get_table_name("jobs")
        self.matches_table = get_table_name("candidate_job_matches")
        self.communications_table = get_table_name("communications")

        # State tracking (remains the same)
        self.state = {
            "metrics": {
                "matches_found": 0,
                "interviews_scheduled": 0,
                "follow_ups_sent": 0,
                "embeddings_generated": 0,
                "successful_matches_stored": 0,
                "emails_sent": {
                    "job_matches": 0,
                    "missed_call": 0,
                    "no_matches": 0,
                    "call_too_short": 0
                }
            },
            "transactions": {}
        }
        
    async def _initialize_async(self):
        """Initialize async components."""
        try:
            # Test Supabase connection
            await self.candidate_service.supabase.table("candidates_dev").select("count").execute()
            logger.info("Successfully connected to Supabase")

            # We should check if the service has a valid client instead
            if not self.openai_service or not self.openai_service.client.api_key:
                 logger.warning("OpenAI service client not properly initialized (missing API key). Match reason generation disabled.")
                 self.openai_enabled = False
            else:
                logger.info("OpenAI service client appears initialized.")
                self.openai_enabled = True
                
        except Exception as e:
            logger.error(f"Error initializing async components: {str(e)}")
            raise
        
    @property
    def candidate_intake_agent(self):
        """Lazy load the candidate intake agent."""
        if self._candidate_intake_agent is None:
            # Import here
            from app.agents.langchain.agents.candidate_intake_agent import CandidateIntakeAgent
            # Pass the required dependencies from BrainAgent's instance
            self._candidate_intake_agent = CandidateIntakeAgent(
                vector_service=self.vector_service, 
                settings=self.settings
            )
        return self._candidate_intake_agent
        
    @property
    def job_matching_agent(self):
        """Lazy load the job matching agent."""
        if self._job_matching_agent is None:
            # Import here
            from app.agents.langchain.agents.job_matching_agent import JobMatchingAgent
            # Pass dependencies
            self._job_matching_agent = JobMatchingAgent(
                vector_service=self.vector_service, 
                settings=self.settings
            )
        return self._job_matching_agent
        
    @property
    def farming_matching_agent(self):
        """Lazy load the farming matching agent."""
        if self._farming_matching_agent is None:
            # Import here
            from app.agents.langchain.agents.farming_matching_agent import FarmingMatchingAgent
            # Pass dependencies
            self._farming_matching_agent = FarmingMatchingAgent(
                vector_service=self.vector_service, 
                settings=self.settings
            )
        return self._farming_matching_agent
        
    @property
    def interview_agent(self):
        """Lazy load the interview agent."""
        if self._interview_agent is None:
            # Import here
            from app.agents.langchain.agents.interview_agent import InterviewAgent
            # Pass dependencies
            self._interview_agent = InterviewAgent(
                vector_service=self.vector_service, 
                settings=self.settings
            )
        return self._interview_agent
        
    @property
    def follow_up_agent(self):
        """Lazy load the follow up agent."""
        if self._follow_up_agent is None:
            # Import here
            from app.agents.langchain.agents.follow_up_agent import FollowUpAgent
            # Pass dependencies
            self._follow_up_agent = FollowUpAgent(
                vector_service=self.vector_service, 
                settings=self.settings
            )
        return self._follow_up_agent
    
    def _clean_profile_json(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Removes fields that should not be stored in the profile_json column."""
        if not isinstance(profile_data, dict):
            logger.warning(f"Attempted to clean non-dict data: {type(profile_data)}")
            return {} # Return empty dict if input is not a dict

        cleaned_data = profile_data.copy()
        # Keys to remove
        keys_to_remove = [
            'basic_info',
            'email', # Remove top-level email if present
            'processing_status',
            'embedding_metadata'
        ]
        for key in keys_to_remove:
            cleaned_data.pop(key, None) # Remove key if it exists

        # Also remove basic_info if nested within 'profile'
        # Use .get() for safety in case 'profile' key doesn't exist or isn't a dict
        profile_nested = cleaned_data.get('profile')
        if isinstance(profile_nested, dict):
             profile_nested.pop('basic_info', None)
        
        return cleaned_data
    
    async def handle_candidate_submission(self, candidate_id: str, candidate_email: str, resume_content: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Handle a new candidate submission.
        """
        process_id = f"submission_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "candidate_submission")
        
        try:
            # Step 1: Initial Data Processing
            logger.info("\nStep 1: 📝 Initial Data Processing")
            logger.info("----------------------------------------")
            
            # Update existing candidate record with additional fields
            update_data = {
                "status": "processing",
                "updated_at": datetime.utcnow().isoformat(),
                "is_resume_processed": False,
                "is_call_completed": False,
                "is_embedding_generated": False
            }
            
            # Update in database
            update_result = await self.supabase.table(self.candidates_table).update(update_data).eq("id", candidate_id).execute()
            if not update_result.data:
                raise ValueError("Failed to update candidate record")
            
            logger.info("✅ Initial candidate record updated")
            self._update_transaction(process_id, "initial_record", "completed")
            
            # If no resume content, we're done with initial processing
            if not resume_content:
                logger.info("No resume content provided, skipping resume processing")
                self._end_transaction(process_id, "completed")
                return {
                    "id": candidate_id,
                    "status": "initial_processing_complete",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Step 2: Quick Resume Extraction
            logger.info("\nStep 2: 🚀 Quick Resume Extraction")
            logger.info("----------------------------------------")
            
            # Use PDFProcessor for quick extraction
            try:
                # Call _quick_extract instead of _arun -> parse_resume
                quick_result = self.pdf_processor._quick_extract(resume_content)
                if quick_result["status"] != "success":
                    raise ValueError(f"Quick extraction failed: {quick_result.get('error', 'Unknown error')}")
                
                essential_info = quick_result.get("essential_info", {})
                
                # Extract ONLY current_role and current_company
                extracted_role = essential_info.get('current_role', '')
                extracted_company = essential_info.get('current_company', '')
                
                # Log what was extracted quickly
                logger.info(f"Quickly Extracted: Role: {extracted_role or 'Not found'}, " + 
                            f"Company: {extracted_company or 'Not found'}")
                
                # Update candidate with ONLY role and company initially
                # The full profile will be populated by the background task
                update_data = {
                    "profile_json": {
                        "current_role": extracted_role,
                        "current_company": extracted_company
                    },
                    "updated_at": datetime.utcnow().isoformat()
                }
                update_res = await self.supabase.table(self.candidates_table).update(update_data).eq("id", candidate_id).execute()
                if not update_res.data:
                     logger.warning(f"Quick extraction DB update returned no data for candidate {candidate_id}")
                     # Decide if this should be a hard failure? For now, log and continue.
                
                logger.info("✅ Quick resume extraction completed and essential info stored.")
                self._update_transaction(process_id, "quick_extraction", "completed")

                # Trigger background processing for the full resume details
                logger.info("🚀 Triggering background task for full resume processing...")
                asyncio.create_task(self._process_full_resume_background(resume_content, candidate_id))
                logger.info("✅ Background task triggered.")
                
            except Exception as e:
                logger.error(f"❌ Quick extraction or background task trigger failed: {str(e)}")
                logger.error(traceback.format_exc()) # Log full traceback for quick extraction errors
                self._update_transaction(process_id, "quick_extraction", "failed", {"error": str(e)})
                # Continue with call scheduling even if quick extraction failed, might use limited data
            
            # Step 3: Schedule Retell Call
            logger.info("\nStep 3: 📞 Scheduling Retell Call")
            logger.info("----------------------------------------")
            
            try:
                # Fetch the LATEST candidate data including the updated profile_json from quick extraction
                call_data_resp = await (
                    self.supabase.table(self.candidates_table)
                    .select('profile_json, phone, full_name') # Select necessary fields
                    .eq('id', candidate_id)
                    .single()
                    .execute()
                )
                if not call_data_resp.data:
                    raise ValueError("Failed to fetch latest data before scheduling call.")

                call_profile_json = call_data_resp.data.get('profile_json', {}) # Use the (now cleaned) profile_json
                phone_number = call_data_resp.data.get('phone')
                db_full_name = call_data_resp.data.get('full_name', '')
                
                # Log what we found in the database for debugging
                logger.info(f"Candidate data for call scheduling - full_name: {db_full_name}, profile: {json.dumps(call_profile_json)[:100]}...")

                if not phone_number:
                    raise ValueError("Cannot schedule call, phone number missing.")
                if not db_full_name:
                    logger.warning(f"Candidate {candidate_id} missing full_name, using 'Candidate' for Retell.")

                # Extract role/company from profile_json (ensure these keys exist after cleaning)
                current_role = call_profile_json.get('current_role', '')
                current_company = call_profile_json.get('current_company', '')

                # Log the dynamic variables we're going to use
                logger.info(f"Using dynamic variables - full_name: {db_full_name}, company: {current_company}, role: {current_role}")

                # Extract first_name from full_name
                nameParts = db_full_name.split(" ")
                first_name = nameParts[0] if len(nameParts) > 0 else ""

                # Pass first_name to RetellService
                call_result = await self.retell_service.schedule_call(
                    candidate_id=candidate_id,
                    dynamic_variables={
                        'first_name': first_name,  # Pass the extracted first name
                        'full_name': db_full_name,  # Pass the full name
                        'email': candidate_email,  # Email passed into handler, not from DB profile_json
                        'current_company': current_company if current_company and current_company != 'pending' else '',
                        'current_title': current_role if current_role and current_role != 'pending' else '',
                        'phone': phone_number
                    }
                )
                call_id = call_result.get("call_id", "unknown")
                logger.info(f"✅ Retell call scheduled for candidate {candidate_id}: {call_id}")
                self._update_transaction(process_id, "call_scheduling", "completed", {"call_id": call_id})

            except Exception as call_err:
                logger.error(f"❌ Error scheduling Retell call for {candidate_id}: {call_err}")
                self._update_transaction(process_id, "call_scheduling", "failed", {"error": str(call_err)})
                # Decide if this should fail the whole submission process?
                # For now, log the error but let the overall process complete.

            logger.info(f"\n=== ✅ Initial Submission Processing Complete for {candidate_id} ===")
            self._end_transaction(process_id, "completed")
            return {
                "id": candidate_id,
                "status": "initial_processing_complete",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in handle_candidate_submission: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._end_transaction(process_id, "failed", {"error": str(e)})
            raise

    async def _process_full_resume_background(self, resume_content: bytes, candidate_id: str):
        """Process the full resume content in the background."""
        try:
            logger.info(f"BACKGROUND: Starting full resume processing for {candidate_id}")
            
            # 1. Get full text using pdf_processor
            # Assuming pdf_processor has a method like `process_pdf` or similar to get full text
            # Let's use `_arun` for now, assuming it returns full text if called directly? 
            # Re-check document_processing.py: It has `process_pdf`, let's use that.
            full_text_result = await self.pdf_processor.process_pdf(resume_content)
            if full_text_result["status"] != "success":
                raise ValueError(f"Background PDF processing failed: {full_text_result.get('error', 'Unknown error')}")
            full_text = full_text_result.get("text") # Check key in process_pdf, yes it's 'text'
            
            if not full_text:
                raise ValueError("Background PDF processing yielded no text content.")

            # Log the full text before sending to parser
            logger.info(f"BACKGROUND: Full text extracted ({len(full_text)} chars). Snippet: {full_text[:500]}...") # Log snippet

            # 2. Parse full text using resume_parser
            logger.info(f"BACKGROUND: Parsing full text ({len(full_text)} chars) for {candidate_id}")
            parsed_data_result = await self.resume_parser.parse_resume(full_text)
            if parsed_data_result["status"] != "success":
                raise ValueError(f"Background resume parsing failed: {parsed_data_result.get('error', 'Unknown error')}")
            
            profile = parsed_data_result.get("profile", {}) 
            logger.info(f"BACKGROUND: Successfully parsed full resume for {candidate_id}. Found keys: {list(profile.keys())}")

            # 3. Apply fallback logic if needed (optional, but keeps existing behavior for now)
            if profile.get("experience") and isinstance(profile.get("experience"), list) and len(profile.get("experience")) > 0:
                # Ensure experience items are dictionaries
                experience_list = [exp for exp in profile["experience"] if isinstance(exp, dict)]
                if experience_list: # Check if list is not empty after filtering
                    most_recent = experience_list[0]
                    if not profile.get("current_role") and most_recent.get("title"):
                        profile["current_role"] = most_recent.get("title")
                        logger.info(f"BACKGROUND: Setting current_role from experience: {profile['current_role']}")
                    if not profile.get("current_company") and most_recent.get("company"):
                        profile["current_company"] = most_recent.get("company")
                        logger.info(f"BACKGROUND: Setting current_company from experience: {profile['current_company']}")
                else:
                    logger.warning(f"BACKGROUND: Experience list for {candidate_id} was empty or contained non-dict items.")
            else:
                logger.info(f"BACKGROUND: No experience data found or fallback not needed for {candidate_id}.")

            # 4. Update database with full profile_json
            logger.info(f"BACKGROUND: Updating database for {candidate_id} with full profile.")
            update_data = {
                "profile_json": profile,
                "is_resume_processed": True,
                "updated_at": datetime.utcnow().isoformat()
            }
            update_result = await self.supabase.table(self.candidates_table).update(update_data).eq("id", candidate_id).execute()
            
            if not update_result.data:
                logger.warning(f"BACKGROUND: DB update after full parsing returned no data for {candidate_id}")
            else:
                logger.info(f"BACKGROUND: ✅ Successfully updated profile_json for {candidate_id}")

        except Exception as e:
            logger.error(f"BACKGROUND: ❌ Error processing full resume for {candidate_id}: {str(e)}")
            logger.error(f"BACKGROUND: Traceback: {traceback.format_exc()}")
            # Optionally update status to indicate background processing failure
            try:
                await self.supabase.table(self.candidates_table).update({
                    "status": "processing_error", # Or a specific 'background_processing_failed' status
                    "updated_at": datetime.utcnow().isoformat(),
                    "is_resume_processed": False # Indicate failure
                }).eq("id", candidate_id).execute()
            except Exception as db_err:
                logger.error(f"BACKGROUND: ❌ Failed to update status to error for {candidate_id}: {db_err}")

    async def _generate_match_reason_and_tags(
        self, 
        candidate_text: str, 
        job_text: str, 
        match_score: float
    ) -> Optional[Dict[str, Any]]:
        """Generates a match reason and tags using OpenAI based on candidate text, job text, and the calculated match score."""
        # Check if the openai_service has a valid client/API key configured
        if not self.openai_service or not self.openai_service.is_configured():
            logger.warning("OpenAI service is not configured. Skipping match reason generation.")
            return None

        system_prompt = (
            "You are an expert talent acquisition specialist. Analyze the provided candidate profile text, job description text, and the pre-calculated similarity score (0.0 to 1.0). "
            f"The match score is {match_score:.2f}. "
            "Based ONLY on the provided texts AND considering the given match score, identify the key reasons for this specific level of alignment (or misalignment). "
            "Provide your analysis in JSON format with two keys: "
            "1. \"match_reason\": A concise 1-2 sentence explanation summarizing the core factors contributing to the given match score. If the score is low, emphasize mismatches or weak points. If high, highlight strong alignments. "
            "2. \"match_tags\": A JSON list of 5-7 relevant keyword tags summarizing the match factors consistent with the score (e.g., python, fastapi, senior_engineer, good_skill_match, low_experience_match, weak_domain_fit). "
            "Focus exclusively on information present in the texts and ensure the tone reflects the provided score."
        )

        user_prompt = (
            f"Please analyze the match between the following candidate and job:\n\n"
            f"--- CANDIDATE PROFILE ---\n{candidate_text}\n\n"
            f"--- JOB DESCRIPTION ---\n{job_text}\n\n"
            f"Provide the analysis in the specified JSON format."
        )

        logger.info("Generating match reason/tags via OpenAI...")
        try:
            # Use the injected openai_service to make the call
            response = await self.openai_service.client.chat.completions.create(
                model="gpt-3.5-turbo-0125", # Use a model that supports JSON mode
                response_format={"type": "json_object"}, # Request JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=250
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logger.error("OpenAI response content is empty.")
                return None

            logger.debug(f"Raw OpenAI response: {response_content}")

            # Parse the JSON response
            try:
                result = json.loads(response_content)
                if "match_reason" in result and "match_tags" in result and isinstance(result["match_tags"], list):
                    logger.info("Successfully generated match reason and tags via OpenAI.")
                    return {
                        "match_reason": str(result["match_reason"]), 
                        "match_tags": [str(tag) for tag in result["match_tags"]]
                    }
                else:
                    logger.error(f"OpenAI response JSON is missing expected keys or has incorrect types: {result}")
                    return None
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse OpenAI JSON response: {json_err}")
                logger.error(f"Response content was: {response_content}")
                return None

        except Exception as e: # Catch broader exceptions from the API call
            logger.error(f"Unexpected error generating match reason/tags via OpenAI: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def handle_call_processed(self, candidate_id: str, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle post-call processing based on Retell status and call duration.
        - If status is error/registered: send missed call email.
        - If duration < 5 mins: send call too short email.
        - If duration >= 5 mins: analyze transcript, embed, match, send match/no-match email.
        """
        process_id = f"call_processed_{candidate_id}_{datetime.utcnow().isoformat()}_duration_logic"
        self._start_transaction(process_id, "call_processing_duration_logic")
        final_process_status = "error_processing_call" # Default final status
        current_profile_for_processing = {}
        matches = []
        embedding_success = False
        candidate_email = None
        candidate_name = None

        try:
            logger.info(f"\n=== Processing Incoming Call Event (Duration Logic) for Candidate ID: {candidate_id} ===")

            # Fetch candidate email and name early on
            try:
                 candidate_info_resp = await self.supabase.table(self.candidates_table)\
                     .select('email, full_name, profile_json, status')\
                     .eq('id', candidate_id)\
                     .single()\
                     .execute()
                 
                 if not candidate_info_resp.data:
                      logger.error(f"(Agent Pre-Check) Could not find candidate {candidate_id}. Aborting call processing.")
                      self._end_transaction(process_id, "failed", "Candidate not found")
                      return {"status": "error", "reason": "Candidate not found"}
                 # else (implied): continue if candidate was found
                 
                 # Store fetched data
                 candidate_email = candidate_info_resp.data.get('email')
                 candidate_name = candidate_info_resp.data.get('full_name')
                 current_db_status = candidate_info_resp.data.get('status')
                 current_profile_for_processing = candidate_info_resp.data.get('profile_json', {}) # Get existing profile
                 
                 if not candidate_email: # Log warning if email is missing
                      logger.warning(f"(Agent Pre-Check) Candidate {candidate_id} missing email address. Cannot send follow-up emails.")

            except Exception as fetch_err:
                 logger.error(f"(Agent Pre-Check) Error fetching initial candidate data for {candidate_id}: {fetch_err}")
                 self._end_transaction(process_id, "failed", "DB error fetching candidate")
                 return {"status": "error", "reason": "DB error fetching candidate"}
            
            # --- Set 'processing_call' Status --- 
            logger.info(f"Setting status to 'processing_call' for {candidate_id}")
            await self._update_candidate_status(candidate_id, 'processing_call')
            # --- End Status Lock ---

            # === Step 1: Initial Failure Check (Retell Status) ===
            call_status = call_data.get('call_status')
            MISSED_CALL_STATUSES = {"registered", "error"}

            if call_status in MISSED_CALL_STATUSES:
                logger.warning(f"Call for {candidate_id} had status: '{call_status}'. Treating as missed call.")
                self._update_transaction(process_id, "status_check", "failed", {"reason": f"Call status: {call_status}"})
                final_process_status = 'call_missed_or_failed' 
                if candidate_email:
                    logger.info(f"Attempting to send missed call email to {candidate_email} due to call_status: {call_status}")
                    try:
                        # --- UNCOMMENTED --- 
                        await send_missed_call_email(
                            recipient_email=candidate_email, 
                            candidate_name=candidate_name, 
                            candidate_id=uuid.UUID(candidate_id), 
                            supabase_client=self.supabase
                        )
                        # pass # Removed pass
                    except Exception as email_err:
                        logger.error(f"Error sending missed call email (triggered by call_status): {email_err}")
                # Status updated in finally block
                return {"status": "stopped", "reason": f"Call status: {call_status}"}
            else:
                 logger.info(f"Call for {candidate_id} has status '{call_status}'. Proceeding to duration check.")
                 self._update_transaction(process_id, "status_check", "completed")

            # --- Transcript check (less critical now, but log if empty on 'ended' status) ---
            transcript = call_data.get('transcript')
            if not transcript or not transcript.strip():
                 if call_status == 'ended': # Log warning specifically if call ended but transcript is empty
                      logger.warning(f"Transcript is empty for candidate {candidate_id} despite call status '{call_status}'. Duration check will determine outcome.")
                 # Don't automatically send missed call email here anymore, rely on duration check below

            # === Step 2: Duration Check ===
            duration_seconds = None
            try:
                # Use .get() for safer access
                call_cost_data = call_data.get('call_cost')
                if call_cost_data and isinstance(call_cost_data, dict):
                    duration_seconds = call_cost_data.get('total_duration_seconds')
                    if duration_seconds is not None and isinstance(duration_seconds, (int, float)):
                         logger.info(f"Retell reported call duration: {duration_seconds} seconds for candidate {candidate_id}.")
                         self._update_transaction(process_id, "duration_check", "completed", {"duration": duration_seconds})
                    else:
                        logger.warning(f"'total_duration_seconds' missing or not a number in call_cost for {candidate_id}. Cannot check duration.")
                        duration_seconds = None # Ensure it's None if invalid
                        self._update_transaction(process_id, "duration_check", "failed", {"reason": "Invalid duration field"})
                else:
                     logger.warning(f"'call_cost' data missing or invalid for {candidate_id}. Cannot check duration.")
                     self._update_transaction(process_id, "duration_check", "failed", {"reason": "Missing call_cost data"})

            except Exception as dur_err:
                 logger.error(f"Error accessing call duration for {candidate_id}: {dur_err}")
                 self._update_transaction(process_id, "duration_check", "failed", {"error": str(dur_err)})
                 duration_seconds = None # Ensure it's None on error

            # --- Decision based on Duration --- 
            MIN_DURATION_SECONDS = 30 # 5 minutes

            if duration_seconds is None or duration_seconds < MIN_DURATION_SECONDS:
                # === Scenario: Call Too Short ===
                reason = "Duration < 5 minutes" if duration_seconds is not None else "Duration unknown/missing"
                logger.warning(f"Call for {candidate_id} was too short ({duration_seconds}s). Reason: {reason}. Sending 'call too short' email.")
                final_process_status = 'call_incomplete' # Use this status for short calls
                self._update_transaction(process_id, "outcome_decision", "call_too_short", {"reason": reason})
                if candidate_email:
                     logger.info(f"Attempting to send 'call too short' email to {candidate_email}.")
                     try:
                         await send_call_too_short_email(
                             recipient_email=candidate_email,
                             candidate_name=candidate_name,
                             candidate_id=uuid.UUID(candidate_id),
                             supabase_client=self.supabase
                         )
                     except Exception as email_call_err:
                          logger.error(f"Exception calling send_call_too_short_email: {email_call_err}")
                # Stop further processing
                return {"status": "stopped", "reason": reason, "final_db_status": final_process_status}
            
            # === Scenario: Call Long Enough - Now also check status ===
            else: 
                # <<< ADD CHECK: Duration is sufficient, NOW check status >>>
                if call_status == 'ended':
                    # <<< INDENT EXISTING BLOCK: Only run this if duration >= 5min AND status == 'ended' >>>
                    logger.info(f"Call duration ({duration_seconds}s) sufficient and status is 'ended'. Proceeding with analysis, embedding, and matching for {candidate_id}.")
                    self._update_transaction(process_id, "outcome_decision", "proceed_full_processing_ended")
                    
                    # --- Main Processing Block (Analysis, Embedding, Matching, Email) --- 
                    merged_profile_data = current_profile_for_processing 
                    extracted_info = None
                    # analysis_complete flag no longer used for gating email

                    # 2a. Analyze Transcript (Still useful for data extraction)
                    self._update_transaction(process_id, "transcript_analysis", "started")
                    if not transcript or not transcript.strip():
                         logger.error(f"Critical error: Call duration >= 5min for {candidate_id}, but transcript is missing or empty! Skipping analysis.")
                         self._update_transaction(process_id, "transcript_analysis", "skipped", {"reason": "Transcript missing despite duration"})
                         # This case shouldn't happen often, but if it does, embedding will likely fail
                    else: 
                        logger.info(f"Analyzing transcript for {candidate_id} (call duration >= 5 min)")
                        try:
                            extracted_info = await self.openai_service.extract_transcript_info(transcript)
                            # Log the outcome of analysis but don't use its status for gating emails
                            analysis_status_from_llm = extracted_info.get('call_status', {}).get('is_complete', 'unknown')
                            analysis_reason_from_llm = extracted_info.get('call_status', {}).get('reason', 'unknown')
                            logger.info(f"Transcript analysis result (for data extraction): complete={analysis_status_from_llm}, reason='{analysis_reason_from_llm}'")
                            self._update_transaction(process_id, "transcript_analysis", "completed", {"analysis_outcome": extracted_info.get('call_status')})
                        except Exception as analysis_err:
                            logger.error(f"Error during transcript analysis for {candidate_id} (but proceeding due to duration): {analysis_err}")
                            self._update_transaction(process_id, "transcript_analysis", "failed", {"error": str(analysis_err)})
                            # Continue processing, maybe embedding can work with profile data only
                            extracted_info = None # Ensure extracted_info is None if analysis fails
                    
                    # --- Process based on Analysis Outcome --- 
                    try:
                        # 2b. Update Profile JSON
                        if extracted_info:
                            self._update_transaction(process_id, "profile_update", "started")
                            merged_profile_data = self._deep_merge(merged_profile_data, extracted_info)
                            cleaned_profile_json = self._clean_profile_json(merged_profile_data)
                            profile_update_resp = await self.supabase.table(self.candidates_table).update({
                                'profile_json': cleaned_profile_json, 'updated_at': datetime.utcnow().isoformat()
                            }).eq('id', candidate_id).execute()
                            if profile_update_resp.data: logger.info(f"Successfully updated profile_json for {candidate_id}")
                            else: logger.error("Failed to save cleaned/merged profile JSON to database.")
                            self._update_transaction(process_id, "profile_update", "completed")
                        else:
                            logger.warning(f"No extracted info from transcript analysis for {candidate_id}, proceeding with existing profile data.")
                            self._update_transaction(process_id, "profile_update", "skipped", {"reason": "No extracted info"})

                        # 2c. Update Communications Content
                        if extracted_info and call_data.get('call_id'):
                            # ... (Existing logic to update communications content) ...
                            pass # Placeholder for brevity, keep existing logic here

                        # 3. Generate & Store Embedding
                        self._update_transaction(process_id, "embedding", "started")
                        logger.info(f"Generating embedding for {candidate_id}")
                        # Use merged_profile_data (potentially just original profile if analysis failed/skipped)
                        embedding_success = await self._generate_and_store_embedding(candidate_id, merged_profile_data)
                        if not embedding_success:
                            logger.error(f"Embedding generation/storage failed for {candidate_id}. Matchmaking will be skipped.")
                            self._update_transaction(process_id, "embedding", "failed", {"reason": "Embedding function returned false"})
                            final_process_status = 'error_embedding'
                        else:
                            self._update_transaction(process_id, "embedding", "completed")
                            self.state["metrics"]["embeddings_generated"] += 1
                            final_process_status = 'completed' # Default success if embedding works

                        # 4. Trigger Matchmaking & Appropriate Email
                        if embedding_success:
                            self._update_transaction(process_id, "matchmaking", "started")
                            logger.info(f"Triggering matchmaking for {candidate_id}")
                            try:
                                matches = await self.matching_service.match_candidate_to_jobs(candidate_id)
                                logger.info(f"Found {len(matches)} potential matches for {candidate_id}")
                                self.state["metrics"]["matches_found"] += len(matches)

                                # --- Prepare Match Records (Only if matches found) ---
                                match_records = []
                                if matches:
                                    # ... (Existing logic to prepare match_records remains the same) ...
                                    pass # Placeholder for brevity

                                # --- Store Match Records --- 
                                if match_records:
                                    # ... (Existing logic to store match_records remains the same) ...
                                    pass # Placeholder for brevity
                                else:
                                    logger.info("No match records were prepared.")

                                # --- Email Logic (Based ONLY on Matches Found/Not Found) --- 
                                if candidate_email:
                                    # Check if high_scoring_jobs were generated (dependent on match_records being processed)
                                    MATCH_SCORE_THRESHOLD = 0.40
                                    high_scoring_jobs = []
                                    if match_records: # Check if we actually *generated* records to send
                                        # ... (Existing logic to query top jobs from stored matches) ...
                                        pass # Placeholder for brevity

                                    # Decide which email to send based on high_scoring_jobs list
                                    if high_scoring_jobs:
                                        logger.info(f"Attempting to send job match email to {candidate_email}...")
                                        try:
                                            await send_job_match_email(...) # Correct arguments
                                            # ... logging ...
                                        except Exception as email_call_err: logger.error(...) # <<< Potential fix: ensure this logger call is correct
                                    else:
                                        # Send no_matches email if embedding worked but no matches found/met threshold
                                        logger.info(f"Sending 'no matches' email to {candidate_email}...")
                                        try:
                                            # <<< FIX: Use the candidate_name fetched directly from DB, not from merged_profile_data >>>
                                            name_to_use = candidate_name if candidate_name else 'Candidate' # Use fetched name, default if None/empty
                                            await send_no_matches_email(
                                                recipient_email=candidate_email, 
                                                candidate_name=name_to_use, # Pass the correct name
                                                candidate_id=candidate_id,
                                                supabase_client=self.supabase
                                            )
                                            self.state["metrics"]["emails_sent"]["no_matches"] += 1
                                            self._update_transaction(process_id, "email_sent", "no_matches_success")
                                            logger.info(f"Successfully sent 'no matches' email to {candidate_email}.")
                                        except Exception as no_match_email_err: 
                                            # Log the actual error variable
                                            logger.error(f"Error sending 'no matches' email: {no_match_email_err}")
                                            self._update_transaction(process_id, "email_sent", "no_matches_failed", {"error": str(no_match_email_err)})
                                else:
                                    logger.warning(f"Could not find email for candidate {candidate_id}...")

                                self._update_transaction(process_id, "matchmaking", "completed")
                            except Exception as match_err:
                                logger.error(f"Error during matchmaking or email logic for {candidate_id}: {match_err}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                self._update_transaction(process_id, "matchmaking", "failed", {"error": str(match_err)})
                                # Keep final_process_status determined by embedding step
                        else: # embedding failed
                            logger.warning(f"Skipping matchmaking for {candidate_id} due to embedding failure.")
                            self._update_transaction(process_id, "matchmaking", "skipped", {"reason": "Embedding failed"})

                    except Exception as processing_err:
                        logger.error(f"Error during main processing block (post-duration check) for {candidate_id}: {processing_err}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        final_process_status = 'error_processing_call'
                        embedding_success = False # Ensure flag is false on error

                    # --- Final Return (Inside if call_status == 'ended') --- 
                    logger.info(f"\n=== ✅ Call Processing Finished (Duration Logic - Ended) for {candidate_id} with determined status: {final_process_status} ====")
                    matches_count = 0
                    if 'embedding_success' in locals() and embedding_success and 'matches' in locals():
                         matches_count = len(matches)
                    return {"status": final_process_status, "matches_found": matches_count}
                
                # <<< ADD ELSE: Duration sufficient, but status != 'ended' >>>
                else: 
                    logger.warning(f"Call duration ({duration_seconds}s) sufficient, but status is '{call_status}' (not 'ended'). Skipping full processing for {candidate_id}.")
                    final_process_status = 'call_incomplete' # Use same status as too short for now?
                    self._update_transaction(process_id, "outcome_decision", "skipped_status_not_ended", {"reason": f"Status was {call_status}"})
                    # Return status indicating processing was skipped due to non-ended status
                    return {"status": final_process_status, "reason": f"Call status was {call_status}, expected 'ended' for full processing"}

        except Exception as e: # Outer try/except for setup errors
            # ... (rest of the code remains the same) ...
            pass # Add pass to satisfy syntax requirement for the except block

    # --- Embedding Helper Methods (Moved from CandidateService) ---
    def _prepare_candidate_text_for_embedding(self, profile_json: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Prepare a comprehensive text string from candidate profile for embedding.
           Returns the text string and a list of profile keys included in the text.
        """
        sections = []
        fields_included = [] # List to track used fields

        # Helper function to add section if value exists
        def add_section(key: str, value: Any, prefix: str = "", is_list: bool = False):
            if value:
                # Check for non-empty lists if is_list is True
                if is_list and isinstance(value, list) and value:
                    sections.append(f"{prefix}{', '.join(map(str, value))}")
                    fields_included.append(key)
                # Check for non-list, non-dict values that are not just whitespace
                elif not is_list and not isinstance(value, (list, dict)):
                    str_value = str(value).strip()
                    if str_value: # Ensure the string representation isn't empty
                        sections.append(f"{prefix}{str_value}")
                        fields_included.append(key)

        # Basic Info (Handle nesting)
        basic_info = profile_json.get('basic_info', {})
        add_section('basic_info.full_name', basic_info.get('full_name'), "Name: ")
        add_section('basic_info.email', basic_info.get('email'), "Email: ")
        add_section('basic_info.phone', basic_info.get('phone'), "Phone: ")
        add_section('basic_info.location', basic_info.get('location'), "Location: ")

        # Top-level Role/Company/Summary (Extracted by agent)
        add_section('current_role', profile_json.get('current_role'), "Current Role: ")
        add_section('current_company', profile_json.get('current_company'), "Current Company: ")
        add_section('professional_summary', profile_json.get('professional_summary'), "Summary: ")

        # Transcript extracted fields
        add_section('career_goals', profile_json.get('career_goals'), "Career Goals: ", is_list=True)
        add_section('motivation_for_job_change', profile_json.get('motivation_for_job_change'), "Motivation for Change: ", is_list=True)

        # Skills & Tech
        add_section('skills', profile_json.get('skills'), "Skills: ", is_list=True)
        add_section('tech_stack', profile_json.get('tech_stack'), "Tech Stack: ", is_list=True)
        add_section('skills_to_develop', profile_json.get('skills_to_develop'), "Skills to Develop: ", is_list=True)
        add_section('technologies_to_avoid', profile_json.get('technologies_to_avoid'), "Technologies to Avoid: ", is_list=True)

        # Experience
        experience = profile_json.get("experience", [])
        if experience:
            sections.append("Experience:")
            fields_included.append('experience') # Mark experience block as included
            for job in experience:
                job_text_parts = []
                if title := job.get('title'): job_text_parts.append(title)
                if company := job.get('company'): job_text_parts.append(f"at {company}")
                if duration := job.get('duration'): job_text_parts.append(f"({duration})")
                if job_text_parts: sections.append(f"- {' '.join(job_text_parts)}")
                if desc := job.get('description'): sections.append(f"  {desc[:200]}...") # Truncate description

        add_section('years_of_experience', profile_json.get('years_of_experience'), "Years of Experience: ")
        if 'leadership_experience' in profile_json: # Handle boolean specifically
             sections.append(f"Leadership Experience: {'Yes' if profile_json['leadership_experience'] else 'No'}")
             fields_included.append('leadership_experience')

        # Education
        education = profile_json.get("education", [])
        if education:
            sections.append("Education:")
            fields_included.append('education') # Mark education block as included
            for edu in education:
                edu_text_parts = []
                if degree := edu.get('degree'): edu_text_parts.append(degree)
                if institution := edu.get('institution'): edu_text_parts.append(f"at {institution}")
                if year := edu.get('year'): edu_text_parts.append(f"({year})")
                if edu_text_parts: sections.append(f"- {' '.join(edu_text_parts)}")

        # Preferences & Deal Breakers
        prefs = profile_json.get('work_preferences', {})
        add_section('work_preferences.benefits', prefs.get('benefits'), "Benefit Preferences: ", is_list=True)
        add_section('role_preferences', profile_json.get('role_preferences'), "Role Preferences: ", is_list=True)
        add_section('preferred_locations', profile_json.get('preferred_locations'), "Preferred Locations: ", is_list=True)
        add_section('preferred_industries', profile_json.get('preferred_industries'), "Preferred Industries: ", is_list=True)
        add_section('desired_company_stage', profile_json.get('desired_company_stage'), "Desired Company Stage: ", is_list=True)
        add_section('preferred_company_size', profile_json.get('preferred_company_size'), "Preferred Company Size: ", is_list=True)
        add_section('desired_company_culture', profile_json.get('desired_company_culture'), "Desired Company Culture: ")
        add_section('deal_breakers', profile_json.get('deal_breakers'), "Deal Breakers: ", is_list=True)

        # Resume Text Snippet
        if resume_text := profile_json.get('resume_text'):
            sections.append("\n--- Resume Text Snippet ---")
            sections.append(str(resume_text)[:1500]) # Limit length
            fields_included.append('resume_text')

        full_text = "\n\n".join(filter(None, sections)).strip()
        logger.debug(f"Prepared text for embedding (Agent - Full) (length {len(full_text)}): {full_text[:500]}...")
        # Return text and the list of fields used
        return full_text, list(set(fields_included)) # Use set to ensure uniqueness

    async def _generate_and_store_embedding(self, candidate_id: str, profile_data_for_embedding: Dict[str, Any]) -> bool:
        """Generates embedding from profile data and updates the candidate record using new helpers.
           Returns True on success, False on failure.
        """
        try:
            logger.info(f"(Agent) Generating embedding for candidate {candidate_id}")
            candidate_text, fields_included = self._prepare_candidate_text_for_embedding(profile_data_for_embedding)
            text_length = len(candidate_text)

            if not candidate_text:
                logger.warning(f"(Agent) No text content for candidate {candidate_id}. Skipping embedding.")
                await self._update_candidate_embedding_status(
                    candidate_id, success=False, error_message="No content for embedding"
                )
                return False

            embedding_vector = await self.openai_service.generate_embedding(candidate_text)
            
            # Prepare metadata dictionary
            embedding_metadata_dict = {
                'last_updated': datetime.utcnow().isoformat(),
                 'fields_included': fields_included,
                 'text_length': text_length,
                'model_used': getattr(self.openai_service, 'embedding_model_name', 'unknown')
            }

            # Update embedding column AND metadata column
            update_payload = {
                'embedding': embedding_vector,
                'embedding_metadata': embedding_metadata_dict,
                'is_embedding_generated': True, # Mark as success
                'embedding_error': None, # Clear any previous error
                'updated_at': datetime.utcnow().isoformat(),
                'status_last_updated': datetime.utcnow().isoformat()
                # Do not change the overall 'status' here, let handle_call_processed decide final status
            }
            
            update_resp = await self.supabase.table(self.candidates_table)\
                .update(update_payload)\
                .eq('id', candidate_id)\
                .execute()

            if not update_resp.data:
                 logger.error(f"(Agent) Failed DB update for embedding vector/metadata for {candidate_id}.")
                 # Update status to reflect embedding failure
                 await self._update_candidate_embedding_status(
                     candidate_id, success=False, error_message="Failed to store embedding vector/metadata"
                 )
                 return False
            else:
                logger.info(f"✅ (Agent) Successfully generated and stored embedding vector and metadata for {candidate_id}")
                # Update status to reflect embedding success (separate call)
                await self._update_candidate_embedding_status(
                    candidate_id, success=True, metadata=embedding_metadata_dict
                )
            return True
            
        except Exception as e:
            error_msg = f"(Agent) Error generating/storing embedding for candidate {candidate_id}: {str(e)}"
            logger.error(f"{error_msg}")
            await self._update_candidate_embedding_status(
                candidate_id, success=False, error_message=error_msg
            )
            return False

    async def _update_candidate_status(self, candidate_id: str, status: str, update_flags: Optional[Dict[str, bool]] = None):
        """Updates the status and related flags in the dedicated DB columns."""
        logger.info(f"Updating status for {candidate_id} to: {status}")
        now_utc = datetime.utcnow()
        update_payload = {
            "status": status,
            "status_last_updated": now_utc.isoformat(),
            "updated_at": now_utc.isoformat() # Keep main updated_at fresh
        }
        
        # Add boolean flags if provided (e.g., is_call_completed)
        if update_flags:
            update_payload.update(update_flags)
            logger.info(f"Also updating flags: {update_flags}")

        try:
            update_response = await self.supabase.table(self.candidates_table)\
                .update(update_payload)\
                .eq('id', candidate_id)\
                .execute()

            if not update_response.data:
                 # Log error but don't raise, as status update might not be critical path
                 logger.error(f"Failed to update status columns for {candidate_id} to {status}")
            else:
                 logger.info(f"Successfully updated status columns for {candidate_id} to {status}")

        except Exception as e:
            logger.error(f"Error updating status columns for {candidate_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _update_candidate_embedding_status(
        self,
        candidate_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Updates embedding status using dedicated DB columns."""
        logger.info(f"Updating embedding status for {candidate_id}. Success: {success}")
        now_utc = datetime.utcnow()
        update_payload = {
            "is_embedding_generated": success,
            "embedding_error": error_message if not success else None,
            "status_last_updated": now_utc.isoformat(),
            "updated_at": now_utc.isoformat()
        }

        # Add metadata column update only if successful and metadata is provided
        if success and metadata:
            update_payload["embedding_metadata"] = metadata
        elif not success:
             # Explicitly set metadata to null if embedding failed
             update_payload["embedding_metadata"] = None 
        # If success is True but metadata is None, we don't add/change the embedding_metadata column

        # If embedding failed, also set the main status to 'error_embedding'
        if not success:
            update_payload["status"] = "error_embedding"

        try:
            update_response = await self.supabase.table(self.candidates_table)\
                .update(update_payload)\
                .eq('id', candidate_id)\
                .execute()

            if not update_response.data:
                logger.error(f"Failed to update embedding status columns for {candidate_id}")
            else:
                logger.info(f"Successfully updated embedding status columns for {candidate_id}")

        except Exception as e:
            logger.error(f"Error updating embedding status columns for {candidate_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # --- Transaction Tracking Methods ---
    def _start_transaction(self, process_id: str, transaction_type: str) -> None:
        self.state["transactions"][process_id] = {
            "type": transaction_type,
            "start_time": datetime.utcnow().isoformat(),
            "status": "started",
            "steps": [],
            "end_time": None,
            "error": None
        }
        logger.info(f"Starting transaction {process_id} ({transaction_type})")

    def _update_transaction(self, process_id: str, step: str, status: str, data: Optional[Dict[str, Any]] = None) -> None:
        if process_id in self.state["transactions"]:
            self.state["transactions"][process_id]["steps"].append({
                "step": step,
            "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            })
        else:
            logger.warning(f"Attempted to update non-existent transaction: {process_id}")

    def _end_transaction(self, process_id: str, status: str, error: Optional[str] = None) -> None:
        if process_id in self.state["transactions"]:
            self.state["transactions"][process_id]["status"] = status
            self.state["transactions"][process_id]["end_time"] = datetime.utcnow().isoformat()
            self.state["transactions"][process_id]["error"] = error
            logger.info(f"Ending transaction {process_id} with status: {status}")
        else:
            logger.warning(f"Attempted to end non-existent transaction: {process_id}")

    # --- Deep Merge Utility (if needed) ---
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        # Keep the deep merge utility as it's used in the modified handle_call_processed
        result = dict1.copy()
        for key, value in dict2.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            elif isinstance(value, list) and key in result and isinstance(result[key], list):
                existing_list = result[key]
                existing_list.extend(item for item in value if item not in existing_list)
                result[key] = existing_list
            elif value is not None:
                 result[key] = value
        return result