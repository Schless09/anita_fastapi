# agents/brain_agent.py
from typing import Dict, Any, Optional, List
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

    async def handle_call_processed(self, candidate_id: str, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle post-call processing: transcript analysis, embedding, matching.
        Only proceeds if a transcript is available.
        """
        process_id = f"call_processed_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "call_processing")
        try:
            logger.info(f"\n=== Processing Completed Call for Candidate ID: {candidate_id} ===")

            # --- Add Status Check --- 
            # Fetch current profile *first* to check status before proceeding
            pre_check_response = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            if not pre_check_response.data:
                logger.error(f"(Agent Pre-Check) Could not find candidate profile for {candidate_id}. Aborting call processing.")
                self._end_transaction(process_id, "failed", "Candidate not found")
                return {"status": "error", "reason": "Candidate not found"}
            
            current_profile_for_check = pre_check_response.data.get('profile_json', {})
            current_status = current_profile_for_check.get('processing_status', {}).get('status')

            final_statuses = ['completed', 'call_missed_or_failed', 'error_processing_call']
            if current_status in final_statuses:
                 logger.info(f"(Agent Check) Candidate {candidate_id} already has final processing status '{current_status}'. Skipping call processing for this trigger.")
                 self._end_transaction(process_id, "skipped", f"Already processed with status: {current_status}")
                 return {"status": "skipped", "reason": f"Already processed: {current_status}"} # Exit early
            logger.info(f"(Agent Check) Candidate {candidate_id} status is '{current_status}'. Proceeding with call processing.")
            # --- End Status Check ---

            # 1. Check for Transcript
            transcript = call_data.get('transcript')
            if not transcript or not transcript.strip():
                logger.warning(f"No valid transcript found for call {call_data.get('call_id')} (Candidate: {candidate_id}). Updating status and stopping.")
                self._update_transaction(process_id, "transcript_check", "failed", {"reason": "No transcript"})
                await self._update_candidate_status(candidate_id, 'call_missed_or_failed')
                self._end_transaction(process_id, "stopped")
                return {"status": "stopped", "reason": "No transcript"}
            logger.info(f"Transcript found for call {call_data.get('call_id')}. Proceeding.")
            self._update_transaction(process_id, "transcript_check", "completed")

            # 2. Analyze Transcript & Merge Profile
            self._update_transaction(process_id, "profile_update", "started")
            logger.info(f"Analyzing transcript and merging profile for {candidate_id}")
            try:
                # Fetch existing profile
                fetch_resp = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
                if not fetch_resp.data:
                    raise ValueError(f"Candidate {candidate_id} not found during call processing.")
                current_profile = fetch_resp.data.get('profile_json', {})

                # Extract info from transcript
                extracted_info = await self.openai_service.extract_transcript_info(transcript)
                if not extracted_info:
                    logger.warning(f"No information extracted from transcript for {candidate_id}. Profile will not be updated with transcript data.")
                    merged_profile = current_profile
                else:
                    merged_profile = self._deep_merge(current_profile, extracted_info)
                    logger.info(f"Successfully merged transcript info for {candidate_id}")

                # Update processing status
                if 'processing_status' not in merged_profile: merged_profile['processing_status'] = {}
                merged_profile['processing_status'].update({
                    'status': 'completed',
                    'call_completed': True,
                    'last_updated': datetime.utcnow().isoformat()
                })

                # Save merged profile
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
                await self._update_candidate_status(candidate_id, 'error_processing_call')
                self._end_transaction(process_id, "error")
                raise profile_err

            # 3. Generate & Store Embedding
            self._update_transaction(process_id, "embedding", "started")
            logger.info(f"Generating embedding for {candidate_id}")
            embedding_success = await self._generate_and_store_embedding(candidate_id, merged_profile)
            if not embedding_success:
                logger.error(f"Embedding generation/storage failed for {candidate_id}. Stopping.")
                self._update_transaction(process_id, "embedding", "failed", {"reason": "Embedding function returned false"})
                self._end_transaction(process_id, "failed")
                return {"status": "failed", "reason": "Embedding generation failed"}
            self._update_transaction(process_id, "embedding", "completed")
            self.state["metrics"]["embeddings_generated"] += 1

            # 4. Trigger Matchmaking
            self._update_transaction(process_id, "matchmaking", "started")
            logger.info(f"Triggering matchmaking for {candidate_id}")
            try:
                matches = await self.matching_service.match_candidate_to_jobs(candidate_id)
                logger.info(f"Found {len(matches)} potential matches for {candidate_id}")
                self.state["metrics"]["matches_found"] += len(matches)

                if matches:
                    # Store matches in the database
                    match_records = []
                    for match in matches:
                        job_id = match.get("job_id")
                        similarity_score = match.get("similarity")
                        if job_id is None or similarity_score is None:
                            logger.warning(f"Skipping match storage for candidate {candidate_id} due to missing job_id/similarity: {match}")
                            continue
                        match_records.append({
                            "candidate_id": candidate_id,
                            "job_id": job_id,
                            "match_score": similarity_score,
                            "created_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat()
                        })
                    if match_records:
                        # Log the records being inserted for debugging
                        logger.debug(f"Attempting to insert match records: {match_records}") 
                        insert_response = await self.supabase.table('candidate_job_matches').insert(match_records).execute()
                        if insert_response.data:
                            logger.info(f"✅ Successfully stored {len(insert_response.data)} job matches for candidate {candidate_id}")
                            self.state["metrics"]["successful_matches_stored"] += len(insert_response.data)
                        else:
                            logger.error(f"Failed to store job matches for candidate {candidate_id}. Response: {insert_response}")
                            self._update_transaction(process_id, "match_storage", "failed", {"error": "DB insert failed"})

                self._update_transaction(process_id, "matchmaking", "completed")
            except Exception as match_err:
                logger.error(f"Error during matchmaking or storing matches for {candidate_id}: {match_err}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._update_transaction(process_id, "matchmaking", "failed", {"error": str(match_err)})

            logger.info(f"\n=== ✅ Call Processing Complete for {candidate_id} ===")
            self._end_transaction(process_id, "completed")
            return {"status": "completed", "matches_found": len(matches)}

        except Exception as e:
            error_msg = f"Unhandled error during call processing for {candidate_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            await self._update_candidate_status(candidate_id, 'error_processing_call')
            self._update_transaction(process_id, "error", "failed", {"error": error_msg})
            self._end_transaction(process_id, "error")
            return {"status": "error", "error": error_msg}

    # --- Embedding Helper Methods (Moved from CandidateService) ---
    def _prepare_candidate_text_for_embedding(self, profile_json: Dict[str, Any]) -> str:
        """Prepare a comprehensive text string from candidate profile for embedding."""
        sections = []
        
        # Basic Info (Handle nesting)
        basic_info = profile_json.get('basic_info', {})
        if name := basic_info.get('full_name'): sections.append(f"Name: {name}")
        if email := basic_info.get('email'): sections.append(f"Email: {email}") # Add email?
        if phone := basic_info.get('phone'): sections.append(f"Phone: {phone}") # Add phone?
        if loc := basic_info.get('location'): sections.append(f"Location: {loc}") # Add location?
        
        # Top-level Role/Company (Extracted by agent)
        if role := profile_json.get('current_role'): sections.append(f"Current Role: {role}")
        if company := profile_json.get('current_company'): sections.append(f"Current Company: {company}")
        if summary := profile_json.get('professional_summary'): sections.append(f"Summary: {summary}")
        
        # Transcript extracted fields (may overlap/augment)
        if goals := profile_json.get('career_goals'): sections.append(f"Career Goals: {', '.join(goals)}")
        if motivation := profile_json.get('motivation_for_job_change'): sections.append(f"Motivation for Change: {', '.join(motivation)}")

        # Skills & Tech
        if skills := profile_json.get('skills'): sections.append(f"Skills: {', '.join(skills)}")
        if tech := profile_json.get('tech_stack'): sections.append(f"Tech Stack: {', '.join(tech)}")
        if develop := profile_json.get('skills_to_develop'): sections.append(f"Skills to Develop: {', '.join(develop)}")
        if avoid := profile_json.get('technologies_to_avoid'): sections.append(f"Technologies to Avoid: {', '.join(avoid)}")

        # Experience
        experience = profile_json.get("experience", [])
        if experience:
            sections.append("Experience:")
            for job in experience:
                job_desc = f"- {job.get('title')} at {job.get('company')}"
                if duration := job.get('duration'): job_desc += f" ({duration})"
                sections.append(job_desc)
                if desc := job.get('description'): sections.append(f"  {desc[:200]}...") # Truncate description
        # Use inferred years if available
        if yrs_exp := profile_json.get('years_of_experience'): sections.append(f"Years of Experience: {yrs_exp}") 
        if leadership := profile_json.get('leadership_experience'): sections.append(f"Leadership Experience: {'Yes' if leadership else 'No'}")

        # Education
        education = profile_json.get("education", [])
        if education:
            sections.append("Education:")
            for edu in education:
                edu_desc = f"- {edu.get('degree')} at {edu.get('institution')}"
                if year := edu.get('year'): edu_desc += f" ({year})"
                sections.append(edu_desc)

        # Preferences & Deal Breakers (Add relevant ones from transcript analysis)
        if prefs := profile_json.get('work_preferences', {}):
            if benefits := prefs.get('benefits'): sections.append(f"Benefit Preferences: {', '.join(benefits)}")
            # Add other work prefs if needed
        if roles := profile_json.get('role_preferences'): sections.append(f"Role Preferences: {', '.join(roles)}")
        if locs := profile_json.get('preferred_locations'): sections.append(f"Preferred Locations: {', '.join(locs)}")
        if inds := profile_json.get('preferred_industries'): sections.append(f"Preferred Industries: {', '.join(inds)}")
        if stage := profile_json.get('desired_company_stage'): sections.append(f"Desired Company Stage: {', '.join(stage)}")
        if size := profile_json.get('preferred_company_size'): sections.append(f"Preferred Company Size: {', '.join(size)}")
        if culture := profile_json.get('desired_company_culture'): sections.append(f"Desired Company Culture: {culture}")
        if breakers := profile_json.get('deal_breakers'): sections.append(f"Deal Breakers: {', '.join(breakers)}")

        # Resume Text (if available)
        if resume_text := profile_json.get('resume_text'):
            sections.append("\n--- Resume Text Snippet ---")
            sections.append(str(resume_text)[:1500]) # Limit length

        full_text = "\n\n".join(filter(None, sections)).strip()
        logger.debug(f"Prepared text for embedding (Agent - Full) (length {len(full_text)}): {full_text[:500]}...")
        return full_text

    async def _generate_and_store_embedding(self, candidate_id: str, profile_json_for_embedding: Dict[str, Any]) -> bool:
        """Generates embedding from profile data and updates the candidate record.
           Returns True on success, False on failure.
        """
        try:
            logger.info(f"(Agent) Generating embedding for candidate {candidate_id}")
            candidate_text = self._prepare_candidate_text_for_embedding(profile_json_for_embedding)

            if not candidate_text:
                logger.warning(f"(Agent) No text content for candidate {candidate_id}. Skipping embedding.")
                await self._update_candidate_embedding_status(candidate_id, False, "No content for embedding")
                return False

            embedding = await self.openai_service.generate_embedding(candidate_text)

            success = await self._update_candidate_embedding_status(candidate_id, True, None, embedding)
            if success:
                logger.info(f"✅ (Agent) Successfully generated and stored embedding for {candidate_id}")
                return True
            else:
                logger.error(f"(Agent) Failed attempt to store embedding for {candidate_id}.")
                return False

        except Exception as e:
            error_msg = f"Error generating/storing embedding for candidate {candidate_id}: {str(e)}"
            logger.error(f"(Agent) {error_msg}")
            await self._update_candidate_embedding_status(candidate_id, False, error_msg)
            return False

    async def _update_candidate_embedding_status(self, candidate_id: str, generated: bool, error_msg: Optional[str], embedding: Optional[List[float]] = None):
        """ Helper to update embedding status and optionally the embedding vector. """
        try:
            fetch_resp = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            if not fetch_resp.data:
                logger.error(f"(Agent) Cannot update embedding status; candidate {candidate_id} not found.")
                return False

            profile_json = fetch_resp.data.get('profile_json', {})
            if 'processing_status' not in profile_json: profile_json['processing_status'] = {}

            profile_json['processing_status']['embedding_generated'] = generated
            profile_json['processing_status']['embedding_error'] = error_msg[:200] if error_msg else None
            profile_json['processing_status']['last_updated'] = datetime.utcnow().isoformat()

            update_payload = {
                'profile_json': profile_json,
                'updated_at': datetime.utcnow().isoformat()
            }
            if generated and embedding:
                update_payload['embedding'] = embedding

            update_resp = await self.supabase.table('candidates_dev').update(update_payload).eq('id', candidate_id).execute()

            if not update_resp.data:
                logger.error(f"(Agent) Failed DB update for embedding status for {candidate_id}. Response: {update_resp}")
                return False
            return True
        except Exception as e:
            logger.error(f"(Agent) Exception during embedding status update for {candidate_id}: {e}")
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