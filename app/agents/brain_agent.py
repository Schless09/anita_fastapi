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
from app.services.pinecone_service import PineconeService
from app.services.matching_service import MatchingService
from app.config import get_settings

# Set up logging
import logging
import traceback
logger = logging.getLogger(__name__)

class BrainAgent:
    """
    Main orchestrator agent that coordinates all other agents.
    Uses lazy loading for sub-agents to improve startup time.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreTool,
        candidate_intake_agent: Optional[CandidateIntakeAgent] = None,
        job_matching_agent: Optional[JobMatchingAgent] = None,
        interview_agent: Optional[InterviewAgent] = None,
        follow_up_agent: Optional[FollowUpAgent] = None
    ):
        self._vector_store = vector_store
        self._candidate_intake_agent = candidate_intake_agent
        self._job_matching_agent = job_matching_agent
        self._interview_agent = interview_agent
        self._follow_up_agent = follow_up_agent
        
        logger.info("BrainAgent initialized with vector store")
    
    @property
    def candidate_intake_agent(self) -> CandidateIntakeAgent:
        """Lazy load the candidate intake agent."""
        if self._candidate_intake_agent is None:
            logger.info("Lazy loading CandidateIntakeAgent")
            self._candidate_intake_agent = CandidateIntakeAgent(vector_store=self._vector_store)
        return self._candidate_intake_agent
    
    @property
    def job_matching_agent(self) -> JobMatchingAgent:
        """Lazy load the job matching agent."""
        if self._job_matching_agent is None:
            logger.info("Lazy loading JobMatchingAgent")
            self._job_matching_agent = JobMatchingAgent(vector_store=self._vector_store)
        return self._job_matching_agent
    
    @property
    def interview_agent(self) -> InterviewAgent:
        """Lazy load the interview agent."""
        if self._interview_agent is None:
            logger.info("Lazy loading InterviewAgent")
            self._interview_agent = InterviewAgent(vector_store=self._vector_store)
        return self._interview_agent
    
    @property
    def follow_up_agent(self) -> FollowUpAgent:
        """Lazy load the follow-up agent."""
        if self._follow_up_agent is None:
            logger.info("Lazy loading FollowUpAgent")
            self._follow_up_agent = FollowUpAgent(vector_store=self._vector_store)
        return self._follow_up_agent
    
    async def handle_candidate_submission(self, candidate_data: CandidateCreate) -> Dict[str, Any]:
        """
        Handle a new candidate submission by coordinating between agents.
        """
        try:
            logger.info(f"Processing candidate submission for {candidate_data.email}")
            
            # Process candidate with intake agent
            profile = await self.candidate_intake_agent.process_candidate(candidate_data)
            logger.info(f"✅ Candidate profile created for {candidate_data.email}")
            
            # Find job matches
            matches = await self.job_matching_agent.find_matches(profile)
            logger.info(f"Found {len(matches)} potential job matches")
            
            if matches:
                # Schedule interview if matches found
                interview = await self.interview_agent.schedule_interview(
                    candidate_id=profile['id'],
                    job_matches=matches
                )
                logger.info(f"Interview scheduled for candidate {profile['id']}")
                
                # Set up follow-up communication
                await self.follow_up_agent.setup_follow_up(
                    candidate_id=profile['id'],
                    interview_id=interview['id']
                )
                logger.info(f"Follow-up communication set for candidate {profile['id']}")
            
            return {
                "id": profile['id'],
                "status": "processing_complete",
                "matches": len(matches) if matches else 0,
                "interview_scheduled": bool(matches),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing candidate {candidate_data.email}: {str(e)}")
            raise

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