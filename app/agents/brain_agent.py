# agents/brain_agent.py
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from app.agents.langchain.agents.candidate_intake_agent import CandidateIntakeAgent
from app.agents.langchain.agents.job_matching_agent import JobMatchingAgent
from app.agents.langchain.agents.farming_matching_agent import FarmingMatchingAgent
from app.agents.langchain.agents.interview_agent import InterviewAgent
from app.agents.langchain.agents.follow_up_agent import FollowUpAgent
from app.agents.langchain.tools.vector_store import VectorStoreTool
from app.agents.langchain.tools.document_processing import PDFProcessor, ResumeParser
from app.agents.langchain.tools.matching import MatchingTool
from app.agents.langchain.tools.communication import EmailTool

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
logger = logging.getLogger(__name__)

class BrainAgent:
    def __init__(
        self,
        vector_store: Optional[VectorStoreTool] = None,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize services
        self.candidate_service = CandidateService()
        self.job_service = JobService()
        self.retell_service = RetellService()
        self.openai_service = OpenAIService()
        self.pinecone_service = PineconeService()
        self.matching_service = MatchingService()
        
        # Initialize tools
        self.vector_store = vector_store or VectorStoreTool()
        self.email_tool = EmailTool()
        self.matching_tool = MatchingTool()
        self.pdf_processor = PDFProcessor()
        self.resume_parser = ResumeParser()
        
        # Initialize chains
        self.candidate_processing_chain = CandidateProcessingChain()
        self.job_matching_chain = JobMatchingChain()
        self.interview_scheduling_chain = InterviewSchedulingChain()
        self.follow_up_chain = FollowUpChain()
        
        # Initialize agents
        self.candidate_intake_agent = CandidateIntakeAgent()
        self.job_matching_agent = JobMatchingAgent()
        self.farming_matching_agent = FarmingMatchingAgent()
        self.interview_agent = InterviewAgent()
        self.follow_up_agent = FollowUpAgent()
        
        # Initialize state management
        self.state = {
            "processes": {},  # Track ongoing processes
            "transactions": {},  # Track multi-step transactions
            "errors": {},  # Track errors by process
            "metrics": {  # Track basic metrics
                "candidates_processed": 0,
                "matches_found": 0,
                "interviews_scheduled": 0,
                "follow_ups_sent": 0
            }
        }
        logger.info("BrainAgent initialized successfully")

    def _start_transaction(self, process_id: str, process_type: str) -> None:
        """Start a new transaction for tracking multi-step processes."""
        self.state["transactions"][process_id] = {
            "type": process_type,
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "status": "in_progress",
            "current_step": None
        }

    def _update_transaction(self, process_id: str, step: str, status: str, data: Optional[Dict] = None) -> None:
        """Update transaction status and data."""
        if process_id in self.state["transactions"]:
            self.state["transactions"][process_id]["steps"].append({
                "step": step,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            })
            self.state["transactions"][process_id]["current_step"] = step

    def _end_transaction(self, process_id: str, status: str) -> None:
        """End a transaction with final status."""
        if process_id in self.state["transactions"]:
            self.state["transactions"][process_id].update({
                "end_time": datetime.utcnow().isoformat(),
                "status": status
            })

    async def handle_candidate_submission(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a new candidate submission."""
        process_id = f"candidate_{candidate_data.get('id', datetime.utcnow().isoformat())}"
        self._start_transaction(process_id, "candidate_submission")
        
        try:
            logger.info(f"Processing candidate submission: {process_id}")
            self._update_transaction(process_id, "intake", "in_progress")
            
            result = await self.candidate_intake_agent.process_candidate(candidate_data)
            
            if result["status"] == "success":
                self._update_transaction(process_id, "intake", "completed", result)
                self.state["metrics"]["candidates_processed"] += 1
                self._end_transaction(process_id, "completed")
            else:
                self._update_transaction(process_id, "intake", "failed", result)
                self._end_transaction(process_id, "failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing candidate {process_id}: {str(e)}")
            self._update_transaction(process_id, "intake", "error", {"error": str(e)})
            self._end_transaction(process_id, "error")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
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