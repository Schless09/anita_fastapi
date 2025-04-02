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

from anita.services.email_service import send_job_match_email

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
    
    async def handle_candidate_submission(self, candidate_id: str, candidate_email: str, resume_content: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Handle initial candidate data processing (e.g., resume parsing) via agent.
        Does NOT generate embedding or trigger matching.
        Assumes initial candidate record might already exist.
        """
        process_id = f"initial_submission_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "candidate_submission")
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
                    resume_processed_flag = False
                else:
                    extracted_profile = intake_result["profile"]
                    resume_processed_flag = True
                    logger.info(f"✅ Resume processed successfully for {candidate_id}")
                    self._update_transaction(process_id, "resume_processing", "completed")
            else:
                logger.info(f"No resume content provided for {candidate_id}, skipping agent processing.")
                extracted_profile = {}
                resume_processed_flag = False
                self._update_transaction(process_id, "resume_processing", "skipped")

            # Step 2: Update Profile in Supabase (merge with existing if necessary)
            logger.info(f"\nStep 2: 💾 Updating Profile in DB")
            logger.info("----------------------------------------")
            self._update_transaction(process_id, "profile_update", "started")

            # Fetch existing profile to merge
            existing_profile_resp = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            existing_profile = existing_profile_resp.data.get('profile_json', {}) if existing_profile_resp.data else {}

            # Merge extracted profile with existing, prioritizing extracted data
            merged_profile = existing_profile.copy()
            merged_profile.update(extracted_profile)

            # Ensure processing_status exists and update it
            if 'processing_status' not in merged_profile:
                merged_profile['processing_status'] = {}
            merged_profile['processing_status'].update({
                'resume_processed': resume_processed_flag,
                'last_updated': datetime.utcnow().isoformat(),
                'call_completed': merged_profile['processing_status'].get('call_completed', False),
                'embedding_generated': merged_profile['processing_status'].get('embedding_generated', False),
                'status': merged_profile['processing_status'].get('status', 'submitted')
            })

            update_data = {
                "profile_json": merged_profile,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            update_response = await self.supabase.table('candidates_dev').update(update_data).eq('id', candidate_id).execute()

            if not update_response.data:
                error_msg = f"Failed to update profile in DB for {candidate_id}"
                logger.error(f"❌ {error_msg}")
                self._update_transaction(process_id, "profile_update", "failed", {"error": error_msg})
                self._end_transaction(process_id, "failed")
                raise Exception(error_msg)

            logger.info(f"✅ Candidate {candidate_id} profile updated in Supabase.")
            self._update_transaction(process_id, "profile_update", "completed")

            # Step 3: Schedule Retell Call
            logger.info(f"\nStep 3: 📞 Scheduling Retell Call")
            logger.info("----------------------------------------")
            self._update_transaction(process_id, "call_scheduling", "started")
            try:
                 # Fetch the LATEST profile data AFTER agent processing and DB update
                 latest_profile_resp = await self.supabase.table('candidates_dev').select('profile_json, phone').eq('id', candidate_id).single().execute()
                 if not latest_profile_resp.data:
                      raise ValueError("Failed to fetch latest profile before scheduling call.")

                 latest_profile = latest_profile_resp.data.get('profile_json', {})
                 phone_number = latest_profile_resp.data.get('phone')

                 if not phone_number:
                      raise ValueError("Cannot schedule call, phone number missing from latest fetch.")

                 # Extract details from the LATEST fetched profile, accessing nested full_name
                 full_name = latest_profile.get('basic_info', {}).get('full_name', '') # Correctly access nested name
                 current_role = latest_profile.get('current_role', '') # This should be top-level based on agent prompt
                 current_company = latest_profile.get('current_company', '') # This should be top-level

                 call_result = await self.retell_service.schedule_call(
                      candidate_id=candidate_id,
                      dynamic_variables={
                           # Use latest fetched data with fallbacks
                           'first_name': full_name.split(' ')[0] if full_name else 'Candidate',
                           'email': candidate_email, # Email is passed into the handler
                           'current_company': current_company if current_company and current_company != 'pending' else '',
                           'current_title': current_role if current_role and current_role != 'pending' else '', # Use empty string fallback
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

        logger.info("Attempting to generate match reason/tags via OpenAI...")
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
        final_process_status = "error_processing_call" # Default final status in case of unexpected exit

        try:
            logger.info(f"\n=== Processing Incoming Call Event for Candidate ID: {candidate_id} ===")

            # --- Refined Status Check --- 
            pre_check_response = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            if not pre_check_response.data:
                logger.error(f"(Agent Pre-Check) Could not find candidate profile for {candidate_id}. Aborting call processing.")
                self._end_transaction(process_id, "failed", "Candidate not found")
                return {"status": "error", "reason": "Candidate not found"}
            
            current_profile_for_check = pre_check_response.data.get('profile_json', {})
            current_status = current_profile_for_check.get('processing_status', {}).get('status')

            # Exit if already completed, failed, OR if another process is already working on it
            terminal_or_processing_statuses = ['completed', 'call_missed_or_failed', 'error_processing_call', 'processing_call']
            if current_status in terminal_or_processing_statuses:
                 logger.info(f"(Agent Check) Candidate {candidate_id} status is '{current_status}'. Skipping call processing for this trigger.")
                 self._end_transaction(process_id, "skipped", f"Already processed/processing: {current_status}")
                 return {"status": "skipped", "reason": f"Already processed/processing: {current_status}"} # Exit early
            logger.info(f"(Agent Check) Candidate {candidate_id} status is '{current_status}'. Proceeding.")
            
            # --- Set 'processing' Status IMMEDIATELY ---
            logger.info(f"Setting status to 'processing_call' for {candidate_id}")
            await self._update_candidate_status(candidate_id, 'processing_call')
            # --- End Status Lock ---

            # 1. Check for Transcript
            transcript = call_data.get('transcript')
            if not transcript or not transcript.strip():
                logger.warning(f"No valid transcript found for call {call_data.get('call_id')} (Candidate: {candidate_id}).")
                self._update_transaction(process_id, "transcript_check", "failed", {"reason": "No transcript"})
                final_process_status = 'call_missed_or_failed' # Set final status for finally block
                return {"status": "stopped", "reason": "No transcript"} # Exit before embedding/matching
            logger.info(f"Transcript found for call {call_data.get('call_id')}. Proceeding.")
            self._update_transaction(process_id, "transcript_check", "completed")
            
            # --- Main Processing Block --- 
            # Define merged_profile here before the block
            merged_profile = current_profile_for_check # Start with the profile we already fetched
            embedding_success = False
            matches = []
            
            # 2. Analyze Transcript & Merge Profile (Wrap in try/except)
            self._update_transaction(process_id, "profile_update", "started")
            logger.info(f"Analyzing transcript and merging profile for {candidate_id}")
            try:
                extracted_info = await self.openai_service.extract_transcript_info(transcript)
                if not extracted_info:
                     logger.warning(f"No information extracted from transcript for {candidate_id}. Profile will not be updated with transcript data.")
                else:
                     merged_profile = self._deep_merge(merged_profile, extracted_info)
                     logger.info(f"Successfully merged transcript info for {candidate_id}")

                # Update processing status within the merged profile dictionary
                if 'processing_status' not in merged_profile: merged_profile['processing_status'] = {}
                merged_profile['processing_status'].update({
                     # Keep 'processing_call' until the very end
                     'call_completed': True,
                     'last_updated': datetime.utcnow().isoformat()
                })

                # Save merged profile (without changing final status yet)
                update_resp = await self.supabase.table('candidates_dev').update({
                     'profile_json': merged_profile,
                     'updated_at': datetime.utcnow().isoformat()
                }).eq('id', candidate_id).execute()
                if not update_resp.data:
                     raise Exception("Failed to save merged profile to database.")
                self._update_transaction(process_id, "profile_update", "completed")
            except Exception as profile_err:
                 logger.error(f"Error analyzing transcript or updating profile for {candidate_id}: {profile_err}")
                 self._update_transaction(process_id, "profile_update", "failed", {"error": str(profile_err)})
                 # final_process_status will be set to error_processing_call in outer except/finally
                 raise profile_err # Re-raise to outer handler

            # 3. Generate & Store Embedding
            self._update_transaction(process_id, "embedding", "started")
            logger.info(f"Generating embedding for {candidate_id}")
            # Use the updated merged_profile for embedding generation
            embedding_success = await self._generate_and_store_embedding(candidate_id, merged_profile) 
            if not embedding_success:
                 logger.error(f"Embedding generation/storage failed for {candidate_id}. Matchmaking will be skipped.")
                 self._update_transaction(process_id, "embedding", "failed", {"reason": "Embedding function returned false"})
                 # Let process finish, final status determined later
            else:
                self._update_transaction(process_id, "embedding", "completed")
                self.state["metrics"]["embeddings_generated"] += 1

            # 4. Trigger Matchmaking (Only if embedding succeeded)
            if embedding_success:
                self._update_transaction(process_id, "matchmaking", "started")
                logger.info(f"Triggering matchmaking for {candidate_id}")
                try:
                    matches = await self.matching_service.match_candidate_to_jobs(candidate_id)
                    logger.info(f"Found {len(matches)} potential matches for {candidate_id}")
                    self.state["metrics"]["matches_found"] += len(matches)

                    if matches:
                        self._update_transaction(process_id, "match_storage", "started")
                        match_records = []
                        # Extract all unique job IDs needed
                        job_ids_to_fetch = list(set(match.get("job_id") for match in matches if match.get("job_id") is not None))

                        # Fetch job details (multiple relevant fields) in bulk
                        job_details_map = {}
                        if job_ids_to_fetch:
                            try:
                                # Fetch multiple relevant columns for context
                                job_fields_to_select = [
                                    'id',
                                    'job_title',
                                    'key_responsibilities', # list
                                    'skills_must_have',     # list
                                    'skills_preferred',     # list
                                    'minimum_years_of_experience', # int/str
                                    'seniority', # enum/str
                                    'ideal_candidate_profile', # text
                                    'product_description' # text
                                ]
                                logger.info(f"Fetching details for {len(job_ids_to_fetch)} job IDs: {job_ids_to_fetch}")
                                job_resp = await self.supabase.table('jobs_dev').select(','.join(job_fields_to_select)) \
                                    .in_('id', job_ids_to_fetch).execute()

                                if job_resp.data:
                                    # Store the whole job data dictionary
                                    job_details_map = {job['id']: job for job in job_resp.data}
                                    logger.info(f"Fetched details for {len(job_details_map)} jobs.")
                                else:
                                    logger.warning(f"Could not fetch details for job IDs: {job_ids_to_fetch}. Response: {job_resp}")
                            except Exception as db_err:
                                logger.error(f"Error fetching job details: {db_err}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                # Proceed without job text if fetching fails

                        for match in matches:
                            job_id = match.get("job_id")
                            similarity_score = match.get("similarity")
                            if job_id is None or similarity_score is None:
                                logger.warning(f"Skipping match storage due to missing job_id or score: {match}")
                                continue

                            job_data = job_details_map.get(job_id)
                            job_text = ""
                            if job_data:
                                # Construct job_text from multiple fields
                                parts = []
                                if job_data.get('job_title'): parts.append(f"Job Title: {job_data['job_title']}")
                                if job_data.get('seniority'): parts.append(f"Seniority: {job_data['seniority']}")
                                if job_data.get('minimum_years_of_experience'): parts.append(f"Min Experience: {job_data['minimum_years_of_experience']} years")
                                if job_data.get('key_responsibilities'): parts.append(f"Responsibilities: {', '.join(job_data['key_responsibilities'])}")
                                if job_data.get('skills_must_have'): parts.append(f"Must-Have Skills: {', '.join(job_data['skills_must_have'])}")
                                if job_data.get('skills_preferred'): parts.append(f"Preferred Skills: {', '.join(job_data['skills_preferred'])}")
                                if job_data.get('product_description'): parts.append(f"Product: {job_data['product_description']}")
                                if job_data.get('ideal_candidate_profile'): parts.append(f"Ideal Candidate: {job_data['ideal_candidate_profile']}")
                                job_text = "\n".join(parts)
                            else:
                                logger.warning(f"No job details found for job_id {job_id} in fetched data.")


                            # Generate match reason and tags
                            reason_tags_data = None
                            if job_text and merged_profile:
                                reason_tags_data = await self._generate_match_reason_and_tags(
                                    candidate_text=merged_profile,
                                    job_text=job_text,
                                    match_score=similarity_score
                                )
                            else:
                                logger.warning(f"Skipping reason generation for job {job_id} due to missing candidate or job text.")


                            match_record = {
                                "candidate_id": candidate_id,
                                "job_id": job_id,
                                "match_score": similarity_score,
                                # Add reason and tags if generated
                                "match_reason": reason_tags_data.get("match_reason") if reason_tags_data else None,
                                "match_tags": reason_tags_data.get("match_tags") if reason_tags_data else None
                            }
                            match_records.append(match_record)

                        if match_records:
                            logger.debug(f"Attempting to insert/upsert {len(match_records)} match records into candidate_job_matches_dev")
                            # Use the correct table name with _dev suffix
                            insert_response = await self.supabase.table('candidate_job_matches_dev')\
                                .upsert(match_records, \
                                        on_conflict='candidate_id, job_id', # Specify conflict columns
                                        ignore_duplicates=False # Make sure scores/reasons are updated
                                        ) \
                                .execute()
                            # Check response status
                            # Note: Checking insert_response needs care, Supabase async might differ
                            # Let's assume success if no exception is raised for now, or check status code if available.
                            # Example check (adapt based on actual Supabase async client behavior):
                            if hasattr(insert_response, 'data') or (hasattr(insert_response, 'status_code') and 200 <= insert_response.status_code < 300):
                                logger.info(f"✅ Successfully upserted {len(insert_response.data)} job matches for candidate {candidate_id}")
                                self._update_transaction(process_id, "match_storage", "completed", {"count": len(insert_response.data)})

                                # --- Start Email Logic ---
                                try:
                                    logger.info(f"Checking for high-scoring matches (>0.90) for candidate {candidate_id} to send email.")
                                    
                                    # 1. Get Candidate Email and Name
                                    candidate_info_resp = await self.supabase.table('candidates_dev')\
                                        .select('email', 'full_name')\
                                        .eq('id', candidate_id)\
                                        .limit(1)\
                                        .execute()

                                    candidate_email = None
                                    candidate_name = None
                                    if candidate_info_resp.data:
                                        candidate_email = candidate_info_resp.data[0].get('email')
                                        candidate_name = candidate_info_resp.data[0].get('full_name') # Assuming 'full_name'
                                    
                                    if not candidate_email:
                                        logger.warning(f"Could not find email for candidate {candidate_id}. Cannot send job match email.")
                                    else:
                                        # Define threshold before use
                                        MATCH_SCORE_THRESHOLD = 0.40 # Lowered threshold for testing
                                        
                                        # 2. Get High-Scoring Job Matches (Title and URL) - Using two queries
                                        logger.info(f"Fetching high-scoring match job IDs (>{MATCH_SCORE_THRESHOLD}) for candidate {candidate_id}.")
                                        
                                        # Query 1: Get job IDs with scores above the threshold
                                        matches_ids_resp = await self.supabase.table('candidate_job_matches_dev') \
                                            .select('job_id, match_score') \
                                            .eq('candidate_id', candidate_id) \
                                            .gt('match_score', MATCH_SCORE_THRESHOLD) \
                                            .order('match_score', desc=True) \
                                            .execute()

                                        top_job_ids = []
                                        if matches_ids_resp.data:
                                            # Sort again in Python just to be safe (API order might not be guaranteed?)
                                            sorted_matches = sorted(matches_ids_resp.data, key=lambda x: x.get('match_score', 0), reverse=True)
                                            # Take top 3 job IDs
                                            top_job_ids = [match['job_id'] for match in sorted_matches[:3] if 'job_id' in match]
                                            logger.info(f"Found {len(matches_ids_resp.data)} matches above threshold. Selected top {len(top_job_ids)} job IDs: {top_job_ids}")
                                        else:
                                             logger.info(f"No matches found above threshold {MATCH_SCORE_THRESHOLD} in initial query.")


                                        high_scoring_jobs = []
                                        # Query 2: Get job details for the TOP job IDs
                                        if top_job_ids: # Use the filtered list of IDs
                                            logger.info(f"Fetching details for top job IDs: {top_job_ids}")
                                            jobs_details_resp = await self.supabase.table('jobs_dev') \
                                                .select('id, job_title, job_url') \
                                                .in_('id', top_job_ids) \
                                                .execute()

                                            if jobs_details_resp.data:
                                                # Reconstruct the list with title and url
                                                # We might want to preserve the original order from top_job_ids if Supabase doesn't
                                                job_details_map = {job['id']: job for job in jobs_details_resp.data}
                                                high_scoring_jobs = [
                                                    {'job_title': job_details_map[job_id].get('job_title'), 'job_url': job_details_map[job_id].get('job_url')}
                                                    for job_id in top_job_ids
                                                    if job_id in job_details_map and job_details_map[job_id].get('job_title') and job_details_map[job_id].get('job_url')
                                                ]
                                                logger.info(f"Successfully fetched details for {len(high_scoring_jobs)} top jobs.")
                                            else:
                                                logger.warning(f"Could not fetch details for top job IDs: {top_job_ids}")
                                        
                                        # 3. Send Email if matches exist (logic remains the same)
                                        if high_scoring_jobs:
                                            logger.info(f"Found {len(high_scoring_jobs)} jobs with details matching criteria for candidate {candidate_id}. Attempting to send email to {candidate_email}.")
                                            # Run email sending in background? For now, run synchronously.
                                            email_sent = send_job_match_email(
                                                recipient_email=candidate_email, 
                                                candidate_name=candidate_name, 
                                                job_matches=high_scoring_jobs
                                            )
                                            if email_sent:
                                                logger.info(f"Successfully queued/sent job match email for candidate {candidate_id}.")
                                            else:
                                                logger.error(f"Failed to send job match email for candidate {candidate_id}.")
                                        else:
                                            logger.info(f"No high-scoring matches (>0.90) found for candidate {candidate_id}. Skipping email.")

                                except Exception as email_err:
                                    logger.error(f"Error during email sending logic for candidate {candidate_id}: {email_err}")
                                    logger.error(f"Traceback: {traceback.format_exc()}")
                                    # Log error but don't stop the overall process
                                # --- End Email Logic ---

                            else:
                                logger.error(f"Failed to upsert job matches for candidate {candidate_id}. Response: {insert_response}")
                                self._update_transaction(process_id, "match_storage", "failed", {"error": "DB upsert failed", "response": str(insert_response)})

                    self._update_transaction(process_id, "matchmaking", "completed")
                except Exception as match_err:
                    logger.error(f"Error during matchmaking or storing matches for {candidate_id}: {match_err}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self._update_transaction(process_id, "matchmaking", "failed", {"error": str(match_err)})
                    # Don't set final status to error here, let it be completed if embedding worked
            else:
                 logger.warning(f"Skipping matchmaking for {candidate_id} due to embedding failure.")
                 self._update_transaction(process_id, "matchmaking", "skipped", {"reason": "Embedding failed"})
            
            # If we reached here without a major error being raised
            final_process_status = 'completed'
            logger.info(f"\n=== ✅ Call Processing Workflow Succeeded for {candidate_id} ===")
            return {"status": "completed", "matches_found": len(matches)}
            
        # Outer except block catches errors raised from within the main processing block
        except Exception as e:
            final_process_status = 'error_processing_call' # Ensure status reflects the error
            error_msg = f"Unhandled error during call processing for {candidate_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            self._update_transaction(process_id, "error", "failed", {"error": error_msg})
            return {"status": "error", "error": error_msg}

        finally:
            # --- Ensure final status is set --- 
            logger.info(f"Updating final status to '{final_process_status}' for candidate {candidate_id}.")
            await self._update_candidate_status(candidate_id, final_process_status)
            self._end_transaction(process_id, final_process_status)
            logger.info(f"Transaction {process_id} ended with status: {final_process_status}")
            # --- End Final Status Update ---

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

    async def _generate_and_store_embedding(self, candidate_id: str, profile_json_for_embedding: Dict[str, Any]) -> bool:
        """Generates embedding from profile data and updates the candidate record.
           Returns True on success, False on failure.
        """
        try:
            logger.info(f"(Agent) Generating embedding for candidate {candidate_id}")
            # Get text AND fields used
            candidate_text, fields_included = self._prepare_candidate_text_for_embedding(profile_json_for_embedding)
            text_length = len(candidate_text) # Get length for metadata

            if not candidate_text:
                logger.warning(f"(Agent) No text content for candidate {candidate_id}. Skipping embedding.")
                # Pass empty fields list and 0 length on failure
                await self._update_candidate_embedding_status(candidate_id, False, "No content for embedding", [], 0)
                return False # Indicate failure

            embedding = await self.openai_service.generate_embedding(candidate_text)

            # Update status, passing the embedding vector AND metadata fields
            success = await self._update_candidate_embedding_status(candidate_id, True, None, fields_included, text_length, embedding)

            if success:
                 logger.info(f"✅ (Agent) Successfully generated and stored embedding for {candidate_id}")
                 return True
            else:
                 logger.error(f"(Agent) Failed attempt to store embedding for {candidate_id}.")
                 # Try to update status with failure details, pass metadata that was *attempted*
                 await self._update_candidate_embedding_status(candidate_id, False, "Failed to store embedding", fields_included, text_length)
                 return False # Indicate failure
            
        except Exception as e:
            error_msg = f"Error generating/storing embedding for candidate {candidate_id}: {str(e)}"
            logger.error(f"(Agent) {error_msg}")
            # Try to update status with error, pass empty fields and 0 length
            await self._update_candidate_embedding_status(candidate_id, False, error_msg, [], 0)
            return False # Indicate failure

    async def _update_candidate_embedding_status(self, 
                                                 candidate_id: str, 
                                                 generated: bool, 
                                                 error_msg: Optional[str], 
                                                 fields_included: List[str], # New param
                                                 text_length: int, # New param
                                                 embedding: Optional[List[float]] = None):
        """ Helper to update embedding status and optionally the embedding vector and metadata. """
        try:
            fetch_resp = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            if not fetch_resp.data:
                logger.error(f"(Agent) Cannot update embedding status; candidate {candidate_id} not found.")
                return False

            profile_json = fetch_resp.data.get('profile_json', {})
            if 'processing_status' not in profile_json: profile_json['processing_status'] = {}

            now_iso = datetime.utcnow().isoformat()

            # Update processing status flags
            profile_json['processing_status']['embedding_generated'] = generated
            profile_json['processing_status']['embedding_error'] = error_msg[:200] if error_msg else None
            profile_json['processing_status']['last_updated'] = now_iso

            # Create/Update embedding_metadata
            embedding_metadata = {
                 'last_updated': now_iso,
                 'fields_included': fields_included,
                 'text_length': text_length,
                 'model_used': getattr(self.openai_service, 'embedding_model_name', 'unknown') # Safely get model name
            }
            profile_json['embedding_metadata'] = embedding_metadata

            # Prepare payload for DB update
            update_payload = {
                'profile_json': profile_json,
                'updated_at': now_iso
            }
            # Include embedding vector only on successful generation
            if generated and embedding:
                update_payload['embedding'] = embedding
            
            # Ensure embedding field is set to NULL if generation failed
            # This prevents keeping a stale embedding if regeneration fails
            if not generated:
                 update_payload['embedding'] = None

            update_resp = await self.supabase.table('candidates_dev').update(update_payload).eq('id', candidate_id).execute()

            if not update_resp.data:
                 logger.error(f"(Agent) Failed DB update for embedding status/metadata for {candidate_id}. Response: {update_resp}")
                 return False
            return True
        except Exception as e:
            logger.error(f"(Agent) Exception during embedding status/metadata update for {candidate_id}: {e}")
            return False

    # --- Other Helper Methods ---
    async def _update_candidate_status(self, candidate_id: str, status: str):
        """ Safely update only the processing_status.status field. """
        try:
            fetch_resp = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            if fetch_resp.data:
                profile_json = fetch_resp.data.get('profile_json', {})
                if 'processing_status' not in profile_json: profile_json['processing_status'] = {}
                profile_json['processing_status']['status'] = status
                profile_json['processing_status']['last_updated'] = datetime.utcnow().isoformat()
                await self.supabase.table('candidates_dev').update({
                    'profile_json': profile_json,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', candidate_id).execute()
        except Exception as e:
            logger.error(f"Failed to update candidate status to '{status}' for {candidate_id}: {e}")

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """ Deep merge two dictionaries, handling nested dicts and lists (unique append). """
        result = dict1.copy()
        for key, value in dict2.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively merge dictionaries
                result[key] = self._deep_merge(result[key], value)
            elif isinstance(value, list) and key in result and isinstance(result[key], list):
                # Append unique items from dict2's list to dict1's list
                existing_list = result[key]
                existing_list.extend(item for item in value if item not in existing_list)
                result[key] = existing_list
            elif value is not None:
                # Overwrite or add the value if it's not None
                # Handles cases where key not in result or result[key] is not dict/list
                result[key] = value
            # If value is None, we implicitly keep dict1's value
        return result

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