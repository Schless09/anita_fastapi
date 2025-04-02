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
        memory: Optional[Any] = None
    ):
        super().__init__(model_name, temperature, memory)
        
        # Initialize tools
        vector_store_tool = VectorStoreTool.get_instance()  # Use singleton instance
        self.tools = [
            vector_store_tool,
            MatchingTool(vector_store=vector_store_tool),
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
            # Get job details using vector store tool
            job_result = await self.tools[0]._arun(
                "search_jobs",
                query=f"job_id:{job_id}",
                top_k=1
            )
            
            if job_result["status"] != "success" or not job_result["results"]:
                return {
                    "status": "error",
                    "error": "Job not found"
                }
            
            job_data = job_result["results"][0]
            
            # Find matching candidates using matching tool
            matches_result = await self.tools[1]._arun(
                "match_job_candidates",
                job_id=job_id,
                top_k=top_k,
                min_score=min_score
            )
            
            if matches_result["status"] != "success":
                return matches_result
            
            # Generate match report using LLM
            report = await self.run(
                f"Generate detailed match report for job {job_id} with {len(matches_result['matches'])} matches:\n"
                f"Job: {job_data}\n"
                f"Matches: {matches_result['matches']}"
            )
            
            return {
                "status": "success",
                "job_id": job_id,
                "job_data": job_data,
                "matches": matches_result["matches"],
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