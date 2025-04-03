# agents/brain_agent.py
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from app.agents.langchain.agents.candidate_intake_agent import CandidateIntakeAgent
from app.agents.langchain.agents.job_matching_agent import JobMatchingAgent
from app.agents.langchain.agents.farming_matching_agent import FarmingMatchingAgent
from app.agents.langchain.agents.interview_agent import InterviewAgent
from app.agents.langchain.agents.follow_up_agent import FollowUpAgent
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
from app.config import get_settings
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

from anita.services.email_service import send_job_match_email, send_missed_call_email, send_no_matches_email

logger = logging.getLogger(__name__)

class BrainAgent:
    """Orchestrator agent that coordinates other specialized agents."""
    
    def __init__(self):
        """Initialize the brain agent and required services."""
        self._candidate_intake_agent = None
        self._job_matching_agent = None
        self._farming_matching_agent = None
        self._interview_agent = None
        self._follow_up_agent = None

        # Initialize needed services
        self.supabase = get_supabase_client()
        self.candidate_service = CandidateService()
        self.openai_service = OpenAIService()
        self.matching_service = MatchingService()
        self.retell_service = RetellService()

        # State tracking (remains the same)
        self.state = {
            "metrics": {
                "matches_found": 0,
                "interviews_scheduled": 0,
                "follow_ups_sent": 0,
                "embeddings_generated": 0,
                "successful_matches_stored": 0
            },
            "transactions": {}
        }
        
    async def _initialize_async(self):
        """Initialize async components."""
        try:
            # Test Supabase connection
            await self.candidate_service.supabase.table("candidates_dev").select("count").execute()
            logger.info("Successfully connected to Supabase")

            # Initialize OpenAI client
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables. Match reason generation will be disabled.")
                self.openai_enabled = False
            else:
                self.openai_enabled = True
                # TODO: Consider using async OpenAI client if available and beneficial
                # from openai import AsyncOpenAI
                # self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            logger.error(f"Error initializing async components: {str(e)}")
            raise
        
    @property
    def candidate_intake_agent(self):
        """Lazy load the candidate intake agent."""
        if self._candidate_intake_agent is None:
            self._candidate_intake_agent = CandidateIntakeAgent()
        return self._candidate_intake_agent
        
    @property
    def job_matching_agent(self):
        """Lazy load the job matching agent."""
        if self._job_matching_agent is None:
            self._job_matching_agent = JobMatchingAgent()
        return self._job_matching_agent
        
    @property
    def farming_matching_agent(self):
        """Lazy load the farming matching agent."""
        if self._farming_matching_agent is None:
            self._farming_matching_agent = FarmingMatchingAgent()
        return self._farming_matching_agent
        
    @property
    def interview_agent(self):
        """Lazy load the interview agent."""
        if self._interview_agent is None:
            self._interview_agent = InterviewAgent()
        return self._interview_agent
        
    @property
    def follow_up_agent(self):
        """Lazy load the follow up agent."""
        if self._follow_up_agent is None:
            self._follow_up_agent = FollowUpAgent()
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
        Handle initial candidate data processing (e.g., resume parsing) via agent.
        Does NOT generate embedding or trigger matching.
        Assumes initial candidate record might already exist.
        """
        process_id = f"initial_submission_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "candidate_submission")
        resume_processed_flag = False # Initialize flag
        current_status = 'submitted' # Default initial status
        try:
            logger.info(f"\n=== Processing Initial Submission for Candidate ID: {candidate_id} ===")
            self._update_transaction(process_id, "resume_processing", "started")

            # Step 1: Process Resume with Intake Agent (if resume provided)
            if resume_content:
                logger.info(f"\nStep 1: 📄 Processing Resume")
                logger.info("----------------------------------------")
                intake_result = await self.candidate_intake_agent.process_candidate(
                        resume_content=resume_content,
                        candidate_email=candidate_email,
                        candidate_id=candidate_id
                    )
                if not intake_result or intake_result.get("status") != "success" or "profile" not in intake_result:
                    error_msg = intake_result.get("error", "Failed to process resume")
                    logger.error(f"❌ Resume processing failed for {candidate_id}: {error_msg}")
                    self._update_transaction(process_id, "resume_processing", "failed", {"error": error_msg})
                    extracted_profile = {}
                    # resume_processed_flag remains False
                else:
                    extracted_profile = intake_result["profile"]
                    resume_processed_flag = True
                    logger.info(f"✅ Resume processed successfully for {candidate_id}")
                    self._update_transaction(process_id, "resume_processing", "completed")
            else:
                logger.info(f"No resume content provided for {candidate_id}, skipping agent processing.")
                extracted_profile = {}
                # resume_processed_flag remains False
                self._update_transaction(process_id, "resume_processing", "skipped")

            # Step 2: Update Profile and Status Columns in Supabase
            logger.info(f"\nStep 2: 💾 Updating Profile & Status in DB")
            logger.info("----------------------------------------")
            self._update_transaction(process_id, "profile_update", "started")

            # Fetch existing profile JSON to merge (optional, merge logic can be complex)
            # For simplicity here, we'll assume extracted_profile is the primary source for profile_json content
            # If merging is critical, fetch existing profile_json and merge carefully
            # existing_profile_resp = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            # existing_profile = existing_profile_resp.data.get('profile_json', {}) if existing_profile_resp.data else {}
            # merged_profile = existing_profile.copy()
            # merged_profile.update(extracted_profile) # Prioritize extracted data

            # Clean the extracted profile before saving
            cleaned_profile_json = self._clean_profile_json(extracted_profile) 

            # Prepare data for the new status columns
            now_utc = datetime.utcnow()
            status_update_data = {
                "is_resume_processed": resume_processed_flag,
                "status": current_status, # Set initial status
                "status_last_updated": now_utc.isoformat(),
                 # Keep other flags as default or update if needed based on initial submission logic
                # "is_call_completed": False, 
                # "is_embedding_generated": False,
                # "embedding_error": None, 
                "updated_at": now_utc.isoformat() # Also update the main updated_at
            }

            # Combine data for the update operation
            update_data = {
                "profile_json": cleaned_profile_json,
                **status_update_data # Unpack the status fields
            }
            
            # Perform the update
            update_response = await self.supabase.table('candidates_dev').update(update_data).eq('id', candidate_id).execute()

            if not update_response.data:
                error_msg = f"Failed to update profile/status in DB for {candidate_id}"
                logger.error(f"❌ {error_msg}")
                self._update_transaction(process_id, "profile_update", "failed", {"error": error_msg})
                self._end_transaction(process_id, "failed")
                raise Exception(error_msg)

            logger.info(f"✅ Candidate {candidate_id} profile and status updated in Supabase.")
            self._update_transaction(process_id, "profile_update", "completed")

            # Step 3: Schedule Retell Call (Code remains largely the same, but fetch needed data)
            logger.info(f"\nStep 3: 📞 Scheduling Retell Call")
            logger.info("----------------------------------------")
            self._update_transaction(process_id, "call_scheduling", "started")
            try:
                 # Fetch needed data for the call: phone and full_name directly, profile_json for role/company
                 call_data_resp = await ( 
                     self.supabase.table('candidates_dev')
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

                 if not phone_number:
                      raise ValueError("Cannot schedule call, phone number missing.")
                 if not db_full_name:
                     logger.warning(f"Candidate {candidate_id} missing full_name, using 'Candidate' for Retell.")


                 # Extract role/company from profile_json (ensure these keys exist after cleaning)
                 current_role = call_profile_json.get('current_role', '') 
                 current_company = call_profile_json.get('current_company', '') 

                 call_result = await self.retell_service.schedule_call(
                      candidate_id=candidate_id,
                      dynamic_variables={
                           'first_name': db_full_name.split(' ')[0] if db_full_name else 'Candidate',
                           'email': candidate_email, # Email passed into handler, not from DB profile_json
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
            error_msg = f"Error during initial submission processing for {candidate_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            self._update_transaction(process_id, "error", "failed", {"error": error_msg})
            self._end_transaction(process_id, "error")
            return {
                "id": candidate_id,
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _generate_match_reason_and_tags(
        self, 
        candidate_text: str, 
        job_text: str, 
        match_score: float
    ) -> Optional[Dict[str, Any]]:
        """Generates a match reason and tags using OpenAI based on candidate text, job text, and the calculated match score."""
        if not self.openai_enabled or not openai.api_key:
            logger.warning("OpenAI is not configured. Skipping match reason generation.")
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
            # Use the ChatCompletions endpoint
            response = openai.chat.completions.create(
                # model="gpt-4-turbo-preview", # Consider gpt-4 for higher quality if needed
                model="gpt-3.5-turbo-0125", # Use a model that supports JSON mode
                response_format={"type": "json_object"}, # Request JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5, # Adjust for desired creativity/factuality
                max_tokens=250 # Limit response size
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logger.error("OpenAI response content is empty.")
                return None

            logger.debug(f"Raw OpenAI response: {response_content}")

            # Parse the JSON response
            try:
                result = json.loads(response_content)
                # Basic validation of expected keys
                if "match_reason" in result and "match_tags" in result and isinstance(result["match_tags"], list):
                    logger.info("Successfully generated match reason and tags via OpenAI.")
                    return {
                        "match_reason": str(result["match_reason"]), # Ensure string type
                        "match_tags": [str(tag) for tag in result["match_tags"]] # Ensure list of strings
                    }
                else:
                    logger.error(f"OpenAI response JSON is missing expected keys or has incorrect types: {result}")
                    return None
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse OpenAI JSON response: {json_err}")
                logger.error(f"Response content was: {response_content}")
                return None

        except openai.APIError as api_err:
            logger.error(f"OpenAI API error: {api_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating match reason/tags via OpenAI: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def handle_call_processed(self, candidate_id: str, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle post-call processing: transcript analysis, embedding, matching.
        Only proceeds if a transcript is available AND status isn't already final/processing.
        """
        process_id = f"call_processed_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "call_processing")
        final_process_status = "error_processing_call" # Default final status
        current_profile_for_processing = {} # Initialize
        matches = [] # Initialize matches
        embedding_success = False # Initialize embedding flag
        candidate_email = None # Initialize
        candidate_name = None # Initialize

        try:
            logger.info(f"\n=== Processing Incoming Call Event for Candidate ID: {candidate_id} ===")

            # Fetch candidate email and name early on, needed for potential emails
            try:
                 candidate_info_resp = await self.supabase.table('candidates_dev')\
                     .select('email, full_name, profile_json, status')\
                     .eq('id', candidate_id)\
                     .single()\
                     .execute()
                 
                 if not candidate_info_resp.data:
                      logger.error(f"(Agent Pre-Check) Could not find candidate {candidate_id}. Aborting call processing.")
                      self._end_transaction(process_id, "failed", "Candidate not found")
                      return {"status": "error", "reason": "Candidate not found"}
            
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
            
            # --- Refined Status Check --- 
            terminal_or_processing_statuses = ['completed', 'call_missed_or_failed', 'error_processing_call', 'processing_call', 'error_embedding']
            if current_db_status in terminal_or_processing_statuses:
                 logger.info(f"(Agent Check) Candidate {candidate_id} status is '{current_db_status}'. Skipping call processing trigger.")
                 self._end_transaction(process_id, "skipped", f"(Agent Check) Candidate {candidate_id} status is '{current_db_status}'. Proceeding.")
                 return {"status": "skipped", "reason": f"(Agent Check) Candidate {candidate_id} status is '{current_db_status}'. Proceeding."} 
            logger.info(f"(Agent Check) Candidate {candidate_id} status is '{current_db_status}'. Proceeding.")
            
            # --- Set 'processing_call' Status IMMEDIATELY in DB column ---
            logger.info(f"Setting status to 'processing_call' for {candidate_id}")
            await self._update_candidate_status(candidate_id, 'processing_call')
            # --- End Status Lock ---

            # 1. Check for Transcript
            transcript = call_data.get('transcript')
            if not transcript or not transcript.strip():
                logger.warning(f"No valid transcript for call {call_data.get('call_id')} (Candidate: {candidate_id}).")
                self._update_transaction(process_id, "transcript_check", "failed", {"reason": "No transcript"})
                final_process_status = 'call_missed_or_failed' 
                # Trigger missed call email HERE before returning (if email exists)
                if candidate_email:
                     logger.info(f"Attempting to send missed call email to {candidate_email} for {candidate_id}")
                     await send_missed_call_email(
                          recipient_email=candidate_email, 
                          candidate_name=candidate_name, 
                          candidate_id=uuid.UUID(candidate_id), # Ensure UUID type
                          supabase_client=self.supabase
                     )
                # Status gets updated in finally block
                return {"status": "stopped", "reason": "No transcript"} 
            self._update_transaction(process_id, "transcript_check", "completed")
            
            # --- Main Processing Block --- 
            merged_profile_data = current_profile_for_processing 
            
            # 2. Analyze Transcript & Update Profile/Status
            self._update_transaction(process_id, "profile_update", "started")
            logger.info(f"Analyzing transcript and updating profile/status for {candidate_id}")
            try:
                extracted_info = await self.openai_service.extract_transcript_info(transcript)
                if not extracted_info:
                     logger.warning(f"No information extracted from transcript for {candidate_id}. Profile JSON not updated.")
                     # Still need to mark call as completed
                     call_completed_flag = True
                else:
                     # Deep merge extracted info into the existing profile data
                     merged_profile_data = self._deep_merge(merged_profile_data, extracted_info)
                     logger.info(f"Successfully merged transcript info into profile data for {candidate_id}")

                     # Clean the merged profile JSON before saving
                     cleaned_profile_json = self._clean_profile_json(merged_profile_data)

                     # Update DB: Save cleaned profile JSON only
                     profile_update_resp = await self.supabase.table('candidates_dev').update({
                         'profile_json': cleaned_profile_json,
                     'updated_at': datetime.utcnow().isoformat()
                }).eq('id', candidate_id).execute()

                     if not profile_update_resp.data:
                          logger.error("Failed to save cleaned/merged profile JSON to database.")
                          # Don't raise Exception here, allow process to continue but log error
                          # We can still try embedding with the un-updated profile
                     else:
                         logger.info(f"Successfully updated profile_json for {candidate_id}")
                     
                     call_completed_flag = True # Mark call completed as info was processed
                
                # Update the is_call_completed flag in the database
                if call_completed_flag:
                    await self._update_candidate_status(candidate_id, 'processing_call', update_flags={'is_call_completed': True})
                
                self._update_transaction(process_id, "profile_update", "completed")
            except Exception as profile_err:
                 logger.error(f"Error during transcript analysis or profile update for {candidate_id}: {profile_err}")
                 self._update_transaction(process_id, "profile_update", "failed", {"error": str(profile_err)})
                 # Let the outer error handler catch this and set final status
                 raise profile_err

            # 3. Generate & Store Embedding
            self._update_transaction(process_id, "embedding", "started")
            logger.info(f"Generating embedding for {candidate_id}")
            # Pass the merged data (potentially updated with transcript info)
            embedding_success = await self._generate_and_store_embedding(candidate_id, merged_profile_data)
            if not embedding_success:
                 logger.error(f"Embedding generation/storage failed for {candidate_id}. Matchmaking will be skipped.")
                 self._update_transaction(process_id, "embedding", "failed", {"reason": "Embedding function returned false"})
                 final_process_status = 'error_embedding' # _update_candidate_embedding_status sets this in DB
            else:
                self._update_transaction(process_id, "embedding", "completed")
                self.state["metrics"]["embeddings_generated"] += 1
                final_process_status = 'completed' # Default success if embedding works

            # 4. Trigger Matchmaking & Appropriate Email (Only if embedding succeeded)
            if embedding_success:
                self._update_transaction(process_id, "matchmaking", "started")
                logger.info(f"Triggering matchmaking for {candidate_id}")
                try:
                    matches = await self.matching_service.match_candidate_to_jobs(candidate_id)
                    logger.info(f"Found {len(matches)} potential matches for {candidate_id}")
                    self.state["metrics"]["matches_found"] += len(matches)

                    logger.info(f"DEBUG: About to check matches condition. matches={bool(matches)}")
                    if matches:
                        logger.info("DEBUG: Inside matches block")
                        self._update_transaction(process_id, "match_storage", "started")
                        # Store matches in database...
                        # Prepare candidate text for matching (using merged data) - Moved here
                        candidate_text_for_match, _ = self._prepare_candidate_text_for_embedding(merged_profile_data)
                        
                        # Fetch job details... - Moved here
                        job_ids_to_fetch = list(set(match.get("job_id") for match in matches if match.get("job_id") is not None))
                        job_details_map = {}
                        if job_ids_to_fetch:
                            try:
                                job_fields_to_select = [
                                    'id', 'job_title', 'key_responsibilities', 'skills_must_have',
                                    'skills_preferred', 'minimum_years_of_experience', 'seniority',
                                    'ideal_candidate_profile', 'product_description', 'job_url' # Add job_url
                                ]
                                job_resp = await self.supabase.table('jobs_dev').select(','.join(job_fields_to_select)) \
                                    .in_('id', job_ids_to_fetch).execute()
                                if job_resp.data:
                                    job_details_map = {job['id']: job for job in job_resp.data}
                                else:
                                    logger.warning(f"Could not fetch details for job IDs: {job_ids_to_fetch}")
                            except Exception as db_err:
                                logger.error(f"Error fetching job details: {db_err}")

                        match_records = []
                        for match in matches:
                            job_id = match.get("job_id")
                            similarity_score = match.get("similarity")
                            if job_id is None or similarity_score is None: continue

                            job_data = job_details_map.get(job_id)
                            job_text = ""
                            if job_data:
                                parts = [
                                    f"Job Title: {job_data.get('job_title')}" if job_data.get('job_title') else None,
                                    f"Seniority: {job_data.get('seniority')}" if job_data.get('seniority') else None,
                                    f"Min Experience: {job_data.get('minimum_years_of_experience')} years" if job_data.get('minimum_years_of_experience') is not None else None,
                                    f"Responsibilities: {', '.join(job_data.get('key_responsibilities', []))}" if job_data.get('key_responsibilities') else None,
                                    f"Must-Have Skills: {', '.join(job_data.get('skills_must_have', []))}" if job_data.get('skills_must_have') else None,
                                    f"Preferred Skills: {', '.join(job_data.get('skills_preferred', []))}" if job_data.get('skills_preferred') else None,
                                    f"Product: {job_data.get('product_description')}" if job_data.get('product_description') else None,
                                    f"Ideal Candidate: {job_data.get('ideal_candidate_profile')}" if job_data.get('ideal_candidate_profile') else None
                                ]
                                job_text = "\n".join(filter(None, parts))
                            
                            reason_tags_data = None
                            if job_text and candidate_text_for_match:
                                reason_tags_data = await self._generate_match_reason_and_tags(
                                    candidate_text=candidate_text_for_match,
                                    job_text=job_text,
                                    match_score=similarity_score
                                )

                            match_records.append({
                                "candidate_id": candidate_id,
                                "job_id": job_id,
                                "match_score": similarity_score,
                                "match_reason": reason_tags_data.get("match_reason") if reason_tags_data else None,
                                "match_tags": reason_tags_data.get("match_tags") if reason_tags_data else None
                            })


                        if match_records:
                            logger.info("DEBUG: Inside match_records block")
                            upsert_response = await self.supabase.table('candidate_job_matches_dev')\
                                .upsert(match_records, on_conflict='candidate_id, job_id', ignore_duplicates=False).execute()
                            if hasattr(upsert_response, 'data') or (hasattr(upsert_response, 'status_code') and 200 <= upsert_response.status_code < 300):
                                logger.info(f"✅ Successfully upserted {len(upsert_response.data)} job matches for candidate {candidate_id}")
                                self._update_transaction(process_id, "match_storage", "completed", {"count": len(upsert_response.data)})
                            else:
                                logger.error(f"Failed to upsert job matches for candidate {candidate_id}. Response: {upsert_response}")
                                self._update_transaction(process_id, "match_storage", "failed", {"error": "DB upsert failed"})

                    # Email logic - Correctly indented after the 'if matches:' block finishes
                    logger.info(f"DEBUG: About to start email logic. candidate_email={bool(candidate_email)}")
                    # Add the try block back for email logic
                    try:
                        if candidate_email:
                            logger.info("DEBUG: Inside email logic with valid candidate_email")
                            MATCH_SCORE_THRESHOLD = 0.40
                            # Get top 3 job IDs above threshold
                            matches_ids_resp = await self.supabase.table('candidate_job_matches_dev').select('job_id, match_score').eq('candidate_id', candidate_id).gt('match_score', MATCH_SCORE_THRESHOLD).order('match_score', desc=True).limit(3).execute()
                            top_job_ids = [m['job_id'] for m in matches_ids_resp.data] if matches_ids_resp.data else []

                            # Fix indentation for high_scoring_jobs initialization and the following block
                            high_scoring_jobs = []
                            if top_job_ids:
                                jobs_details_resp = await self.supabase.table('jobs_dev').select('id, job_title, job_url').in_('id', top_job_ids).execute()
                                # Fix indentation for the inner if block
                                if jobs_details_resp.data:
                                    job_details_map = {job['id']: job for job in jobs_details_resp.data}
                                    high_scoring_jobs = [
                                        {'job_title': job_details_map[job_id].get('job_title'), 'job_url': job_details_map[job_id].get('job_url')}
                                        for job_id in top_job_ids if job_id in job_details_map and job_details_map[job_id].get('job_title') and job_details_map[job_id].get('job_url')
                                    ]
                            
                            # Fix indentation for the if high_scoring_jobs block
                            if high_scoring_jobs:
                                logger.info(f"Attempting to send job match email to {candidate_email} for {len(high_scoring_jobs)} jobs.")
                                await send_job_match_email(
                                    recipient_email=candidate_email, 
                                    candidate_name=candidate_name, 
                                    job_matches=high_scoring_jobs,
                                    candidate_id=uuid.UUID(candidate_id), # Ensure UUID 
                                    supabase_client=self.supabase
                                )
                            # Fix indentation for the else block (sending no matches email)
                            else:
                                # No jobs met threshold for the email OR no matches were found initially
                                logger.info(f"No high-scoring jobs found or initial match count was zero. Attempting to send 'no matches' email to {candidate_email}.")
                                await send_no_matches_email(
                                    recipient_email=candidate_email, 
                                    candidate_name=candidate_name, 
                                    candidate_id=uuid.UUID(candidate_id), # Ensure UUID
                                    supabase_client=self.supabase
                                )
                        # Fix indentation for the else block (no candidate email)
                        else:
                            logger.warning(f"Could not find email for candidate {candidate_id} to send post-match email.")
                    # Add the except block for the email try
                    except Exception as email_err:
                        logger.error(f"Error during email sending logic for {candidate_id}: {email_err}")
                        # Don't re-raise, just log the error and continue

                    self._update_transaction(process_id, "matchmaking", "completed")
                # Add the except block for the matchmaking try
                except Exception as match_err:
                    logger.error(f"Error during matchmaking or storing matches for {candidate_id}: {match_err}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self._update_transaction(process_id, "matchmaking", "failed", {"error": str(match_err)})
                    # If matchmaking fails, the overall process status remains as determined by embedding
                    # final_process_status doesn't change here
            else:
                 logger.warning(f"Skipping matchmaking for {candidate_id} due to embedding failure.")
                 self._update_transaction(process_id, "matchmaking", "skipped", {"reason": "Embedding failed"})
                 # final_process_status is already 'error_embedding'
            
            # final_process_status is now 'completed' or 'error_embedding'
            logger.info(f"\n=== ✅ Call Processing Finished for {candidate_id} with status: {final_process_status} ====")
            return {"status": final_process_status, "matches_found": len(matches)}
            
        except Exception as e:
            # Catch errors from profile update or other unhandled exceptions
            final_process_status = 'error_processing_call' # Set specific error status
            error_msg = f"Unhandled error during call processing for {candidate_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            self._update_transaction(process_id, "error", "failed", {"error": error_msg})
            # Status will be updated in finally block
            return {"status": "error", "error": error_msg}

        finally:
            # --- Ensure final status is set in DB --- 
            logger.info(f"Updating final status to '{final_process_status}' for candidate {candidate_id} in finally block.")
            final_flags = {
                'is_call_completed': True # Call was processed (even if transcript was empty or error occurred)
            }
            # Set embedding flag based on the determined final status
            if final_process_status == 'completed':
                 final_flags['is_embedding_generated'] = True 
            elif final_process_status in ['error_embedding', 'error_processing_call', 'call_missed_or_failed']:
                 final_flags['is_embedding_generated'] = False # Ensure it's false if embedding failed or wasn't reached
            
            # Update the final status and relevant flags
            await self._update_candidate_status(candidate_id, final_process_status, update_flags=final_flags)
            self._end_transaction(process_id, final_process_status)
            logger.info(f"Transaction {process_id} ended with status: {final_process_status}")
            # --- End Final Status Update ---

    async def handle_callback_request(self, candidate_id: str) -> bool:
        """Handles a request from a candidate (via link click) to reschedule a call."""
        process_id = f"callback_request_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "callback_request")
        try:
            logger.info(f"Handling callback request for candidate: {candidate_id}")
            
            # 1. Fetch required candidate data (email, phone, full_name)
            self._update_transaction(process_id, "fetch_data", "started")
            try:
                candidate_data_resp = await self.supabase.table('candidates_dev')\
                    .select('email, phone, full_name')\
                    .eq('id', candidate_id)\
                    .single()\
                    .execute()
                
                if not candidate_data_resp.data:
                    logger.error(f"Callback request failed: Candidate {candidate_id} not found.")
                    self._update_transaction(process_id, "fetch_data", "failed", {"error": "Candidate not found"})
                    self._end_transaction(process_id, "failed")
                    return False # Indicate failure to the endpoint handler
                
                candidate_email = candidate_data_resp.data.get('email')
                phone_number = candidate_data_resp.data.get('phone')
                db_full_name = candidate_data_resp.data.get('full_name')
                
                if not phone_number:
                    logger.error(f"Callback request failed: Phone number missing for candidate {candidate_id}.")
                    self._update_transaction(process_id, "fetch_data", "failed", {"error": "Phone number missing"})
                    self._end_transaction(process_id, "failed")
                    return False
                if not candidate_email:
                     logger.warning(f"Callback request: Email missing for candidate {candidate_id}. Proceeding without it.")
                if not db_full_name:
                     logger.warning(f"Callback request: Full name missing for candidate {candidate_id}. Using fallback.")

                self._update_transaction(process_id, "fetch_data", "completed")

            except Exception as db_err:
                logger.error(f"Callback request failed: DB error fetching data for {candidate_id}: {db_err}")
                self._update_transaction(process_id, "fetch_data", "failed", {"error": str(db_err)})
                self._end_transaction(process_id, "failed")
                return False
            
            # 2. Schedule the Retell call using the fetched data
            #    Reusing the logic from handle_candidate_submission's step 3
            logger.info(f"Attempting to schedule new call for candidate {candidate_id} via callback request.")
            self._update_transaction(process_id, "call_scheduling", "started")
            try:
                 call_result = await self.retell_service.schedule_call(
                      candidate_id=candidate_id,
                      dynamic_variables={
                           'first_name': db_full_name.split(' ')[0] if db_full_name else 'Candidate',
                           'email': candidate_email if candidate_email else '', # Pass email if available
                           # Assuming role/company aren't needed/available for a simple callback
                           'current_company': '',
                           'current_title': '',
                           'phone': phone_number
                      }
                 )
                 call_id = call_result.get("call_id", "unknown")
                 logger.info(f"✅ Retell call re-scheduled via callback for {candidate_id}: {call_id}")
                 self._update_transaction(process_id, "call_scheduling", "completed", {"call_id": call_id})
                 self._end_transaction(process_id, "completed")
                 return True # Indicate success to the endpoint handler

            except Exception as call_err:
                 logger.error(f"❌ Error re-scheduling Retell call via callback for {candidate_id}: {call_err}")
                 self._update_transaction(process_id, "call_scheduling", "failed", {"error": str(call_err)})
                 self._end_transaction(process_id, "failed")
                 return False
                 
        except Exception as outer_err:
            # Catch any unexpected errors in the overall handling
            logger.error(f"Unexpected error in handle_callback_request for {candidate_id}: {outer_err}")
            self._update_transaction(process_id, "error", "failed", {"error": str(outer_err)})
            self._end_transaction(process_id, "error")
            return False

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
            
            update_resp = await self.supabase.table('candidates_dev')\
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
            update_response = await self.supabase.table('candidates_dev')\
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
            update_response = await self.supabase.table('candidates_dev')\
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