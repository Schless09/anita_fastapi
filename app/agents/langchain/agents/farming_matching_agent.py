from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
from ..tools.vector_store import VectorStoreTool
from ..tools.matching import MatchingTool
from ..tools.communication import EmailTool

class FarmingMatchingAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory: Optional[Any] = None,
        vector_store: Optional[Any] = None
    ):
        super().__init__(model_name, temperature, memory)
        
        # Initialize tools
        self.tools = [
            VectorStoreTool(vector_store=vector_store) if vector_store else VectorStoreTool(),  # Use provided vector_store or create a new one
            MatchingTool(vector_store=vector_store),  # Pass vector_store to MatchingTool
            EmailTool()
        ]
        
        # Initialize agent with system message
        system_message = """You are an AI recruitment assistant specializing in proactive job matching.
        Your responsibilities include:
        1. Identifying unmatched candidates in the system
        2. Analyzing new job postings
        3. Finding potential matches between new jobs and unmatched candidates
        4. Prioritizing matches based on candidate experience and job requirements
        5. Managing communication for potential matches
        
        Focus on creating meaningful connections between candidates and jobs."""
        
        self._initialize_agent(system_message)

    async def find_unmatched_candidates(
        self,
        days_threshold: int = 30
    ) -> Dict[str, Any]:
        """Find candidates who haven't been matched in recent days."""
        try:
            # Get unmatched candidates
            candidates = await self.run(
                f"Find candidates not matched in the last {days_threshold} days"
            )
            
            return {
                "status": "success",
                "unmatched_candidates": candidates
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def match_new_jobs(
        self,
        job_ids: List[str],
        unmatched_candidates: List[Dict[str, Any]],
        min_score: float = 0.7
    ) -> Dict[str, Any]:
        """Match new jobs to unmatched candidates."""
        try:
            matches = []
            for job_id in job_ids:
                # Get job details
                job_data = await self.run(f"Get job details for job_id: {job_id}")
                
                # Find matches among unmatched candidates
                job_matches = await self.run(
                    f"Find matches for job {job_id} among {len(unmatched_candidates)} unmatched candidates"
                )
                
                # Filter by minimum score
                filtered_matches = [
                    match for match in job_matches
                    if match.get("score", 0) >= min_score
                ]
                
                matches.extend(filtered_matches)
            
            # Sort matches by score
            matches.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return {
                "status": "success",
                "matches": matches
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def notify_potential_matches(
        self,
        matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Notify candidates about potential job matches."""
        try:
            results = []
            for match in matches:
                # Send notification email
                email_result = await self.run(
                    f"Send potential match notification to candidate {match['candidate_id']} "
                    f"for job {match['job_id']}"
                )
                results.append({
                    "candidate_id": match["candidate_id"],
                    "job_id": match["job_id"],
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