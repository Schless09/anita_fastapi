from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
from ..tools.vector_store import VectorStoreTool
from ..tools.matching import MatchingTool
from ..tools.communication import EmailTool

class JobMatchingAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory: Optional[Any] = None,
        vector_store: Optional[VectorStoreTool] = None
    ):
        super().__init__(model_name, temperature, memory)
        
        # Initialize tools
        self.tools = [
            vector_store or VectorStoreTool(),
            MatchingTool(vector_store=vector_store),
            EmailTool()
        ]
        
        # Initialize agent with system message
        system_message = """You are an AI recruitment assistant specializing in job-candidate matching.
        Your responsibilities include:
        1. Analyzing job requirements and candidate profiles
        2. Performing semantic matching between jobs and candidates
        3. Scoring and ranking matches based on relevance
        4. Generating detailed match reports
        5. Handling communication for matched candidates
        
        Always ensure fair and unbiased matching based on skills and experience."""
        
        self._initialize_agent(system_message)

    async def find_matches(
        self,
        job_id: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> Dict[str, Any]:
        """Find matching candidates for a job."""
        try:
            # Get job details
            job_data = await self.run(f"Get job details for job_id: {job_id}")
            
            # Find matching candidates
            matches = await self.run(
                f"Find top {top_k} matching candidates for job: {job_data}"
            )
            
            # Filter by minimum score
            filtered_matches = [
                match for match in matches
                if match.get("score", 0) >= min_score
            ]
            
            # Generate match report
            report = await self.run(
                f"Generate detailed match report for job {job_id} with {len(filtered_matches)} matches"
            )
            
            return {
                "status": "success",
                "job_id": job_id,
                "matches": filtered_matches,
                "report": report
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def notify_matches(
        self,
        job_id: str,
        matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Notify matched candidates."""
        try:
            results = []
            for match in matches:
                # Send notification email
                email_result = await self.run(
                    f"Send match notification to candidate {match['candidate_id']} "
                    f"for job {job_id}"
                )
                results.append({
                    "candidate_id": match["candidate_id"],
                    "email_sent": email_result.get("success")
                })
            
            return {
                "status": "success",
                "notifications_sent": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 