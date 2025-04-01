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

# Set up logging
import logging
import traceback
import uuid
logger = logging.getLogger(__name__)

class BrainAgent:
    """Orchestrator agent that coordinates other specialized agents."""
    
    def __init__(self, vector_store):
        """Initialize the brain agent with a vector store."""
        self.vector_store = vector_store
        self._candidate_intake_agent = None
        self._job_matching_agent = None
        self._interview_agent = None
        self._follow_up_agent = None
        
    @property
    def candidate_intake_agent(self):
        """Lazy load the candidate intake agent."""
        if self._candidate_intake_agent is None:
            self._candidate_intake_agent = CandidateIntakeAgent(vector_store=self.vector_store)
        return self._candidate_intake_agent
        
    @property
    def job_matching_agent(self):
        """Lazy load the job matching agent."""
        if self._job_matching_agent is None:
            self._job_matching_agent = JobMatchingAgent(vector_store=self.vector_store)
        return self._job_matching_agent
        
    @property
    def interview_agent(self):
        """Lazy load the interview agent."""
        if self._interview_agent is None:
            self._interview_agent = InterviewAgent(vector_store=self.vector_store)
        return self._interview_agent
        
    @property
    def follow_up_agent(self):
        """Lazy load the follow up agent."""
        if self._follow_up_agent is None:
            self._follow_up_agent = FollowUpAgent(vector_store=self.vector_store)
        return self._follow_up_agent
    
    async def handle_candidate_submission(self, candidate_data: CandidateCreate) -> Dict[str, Any]:
        """
        Handle a new candidate submission.
        This includes:
        1. Processing the resume
        2. Storing initial profile data
        3. Scheduling the Retell AI call
        Note: Vector store storage happens after call completion
        """
        try:
            process_id = str(uuid.uuid4())
            logger.info(f"\n=== Starting Candidate Processing (ID: {process_id}) ===")
            logger.info(f"Processing candidate: {candidate_data.email}")
            
            # Step 1: Process Resume
            logger.info(f"\nStep 1: 📄 Processing Resume")
            logger.info("----------------------------------------")
            
            # Process resume with candidate intake agent
            intake_result = await self.candidate_intake_agent.process_candidate(
                resume_content=candidate_data.resume_content,
                candidate_email=candidate_data.email,
                candidate_id=candidate_data.id
            )
            
            if intake_result["status"] != "success":
                logger.error(f"❌ Resume Processing Failed: {intake_result}")
                return intake_result
            
            # Step 2: Update Supabase with processed candidate data
            logger.info(f"\nStep 2: 📊 Updating Supabase with processed data")
            logger.info("----------------------------------------")
            
            # Create candidate service
            candidate_service = CandidateService()
            
            # Prepare update data
            update_data = {
                "profile_json": {
                    **intake_result["profile"],
                    "processing_status": {
                        "resume_processed": True,
                        "call_completed": False,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                },
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Update in Supabase
            updated_candidate = await candidate_service.update_candidate_profile(
                candidate_id=candidate_data.id,
                update_data=update_data
            )
            logger.info(f"✅ Candidate {candidate_data.id} updated in Supabase with processed resume data")
            
            # Step 3: Schedule Retell AI Call
            logger.info(f"\nStep 3: 📞 Scheduling Retell AI Call")
            logger.info("----------------------------------------")
            
            # Schedule call with Retell
            await candidate_service.schedule_initial_contact(candidate_data.id)
            logger.info(f"✅ Retell AI call scheduled for candidate {candidate_data.id}")
            
            logger.info(f"\n=== ✅ Initial processing complete (ID: {process_id}) ===")
            
            return {
                "id": candidate_data.id,
                "status": "processing_complete",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing candidate {candidate_data.email}: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "id": candidate_data.id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def handle_job_matching(self, job_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find matching candidates for a job."""
        process_id = f"job_matching_{job_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "job_matching")
        
        try:
            logger.info(f"Finding matches for job: {job_id}")
            self._update_transaction(process_id, "matching", "in_progress")
            
            result = await self.job_matching_agent.find_matches(job_id, top_k)
            
            if result["status"] == "success":
                self._update_transaction(process_id, "matching", "completed", result)
                self.state["metrics"]["matches_found"] += len(result.get("matches", []))
                self._end_transaction(process_id, "completed")
            else:
                self._update_transaction(process_id, "matching", "failed", result)
                self._end_transaction(process_id, "failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding matches for job {job_id}: {str(e)}")
            self._update_transaction(process_id, "matching", "error", {"error": str(e)})
            self._end_transaction(process_id, "error")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }

    async def handle_farming_matching(self, candidate_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find matching jobs for a candidate."""
        process_id = f"farming_{candidate_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "farming_matching")
        
        try:
            logger.info(f"Finding jobs for candidate: {candidate_id}")
            self._update_transaction(process_id, "farming", "in_progress")
            
            result = await self.farming_matching_agent.find_matches(candidate_id, top_k)
            
            if result["status"] == "success":
                self._update_transaction(process_id, "farming", "completed", result)
                self.state["metrics"]["matches_found"] += len(result.get("matches", []))
                self._end_transaction(process_id, "completed")
            else:
                self._update_transaction(process_id, "farming", "failed", result)
                self._end_transaction(process_id, "failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding jobs for candidate {candidate_id}: {str(e)}")
            self._update_transaction(process_id, "farming", "error", {"error": str(e)})
            self._end_transaction(process_id, "error")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }

    async def handle_interview_scheduling(
        self,
        candidate_id: str,
        job_id: str,
        preferred_times: List[str]
    ) -> Dict[str, Any]:
        """Schedule an interview for a candidate."""
        process_id = f"interview_{candidate_id}_{job_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "interview_scheduling")
        
        try:
            logger.info(f"Scheduling interview for candidate {candidate_id} and job {job_id}")
            self._update_transaction(process_id, "scheduling", "in_progress")
            
            result = await self.interview_agent.schedule_interview(
                candidate_id,
                job_id,
                preferred_times
            )
            
            if result["status"] == "success":
                self._update_transaction(process_id, "scheduling", "completed", result)
                self.state["metrics"]["interviews_scheduled"] += 1
                self._end_transaction(process_id, "completed")
            else:
                self._update_transaction(process_id, "scheduling", "failed", result)
                self._end_transaction(process_id, "failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scheduling interview for candidate {candidate_id}: {str(e)}")
            self._update_transaction(process_id, "scheduling", "error", {"error": str(e)})
            self._end_transaction(process_id, "error")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }

    async def handle_follow_up(
        self,
        interaction_id: str,
        interaction_type: str
    ) -> Dict[str, Any]:
        """Handle follow-up for an interaction."""
        process_id = f"follow_up_{interaction_id}_{datetime.utcnow().isoformat()}"
        self._start_transaction(process_id, "follow_up")
        
        try:
            logger.info(f"Processing follow-up for interaction: {interaction_id}")
            self._update_transaction(process_id, "follow_up", "in_progress")
            
            result = await self.follow_up_agent.process_follow_up(
                interaction_id,
                interaction_type
            )
            
            if result["status"] == "success":
                self._update_transaction(process_id, "follow_up", "completed", result)
                self.state["metrics"]["follow_ups_sent"] += 1
                self._end_transaction(process_id, "completed")
            else:
                self._update_transaction(process_id, "follow_up", "failed", result)
                self._end_transaction(process_id, "failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing follow-up for interaction {interaction_id}: {str(e)}")
            self._update_transaction(process_id, "follow_up", "error", {"error": str(e)})
            self._end_transaction(process_id, "error")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "status": "success",
            "metrics": self.state["metrics"],
            "active_processes": len([t for t in self.state["transactions"].values() if t["status"] == "in_progress"])
        }

    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """Get status of a specific process."""
        if process_id in self.state["transactions"]:
            return {
                "status": "success",
                "process": self.state["transactions"][process_id]
            }
        return {
            "status": "error",
            "error": f"Process {process_id} not found"
        }

    async def check_call_status(self, candidate_id: str, call_id: str) -> Dict[str, Any]:
        """Check the status of a call for a candidate."""
        try:
            logger.info(f"Checking call status for candidate {candidate_id}, call {call_id}")
            
            # Use the retell service to check the call status
            status = await self.retell_service.get_call_status(call_id)
            
            # Store the status check as a transaction
            process_id = f"call_status_{call_id}_{datetime.utcnow().isoformat()}"
            self._start_transaction(process_id, "call_status_check")
            self._update_transaction(process_id, "status_check", "completed", {"status": status})
            self._end_transaction(process_id, "completed")
            
            return {
                "status": "success",
                "call_status": status,
                "candidate_id": candidate_id
            }
            
        except Exception as e:
            logger.error(f"Error checking call status for candidate {candidate_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }