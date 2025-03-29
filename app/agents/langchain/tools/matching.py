from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from .vector_store import VectorStoreTool
from pydantic import Field
from .base import parse_llm_json_response

class MatchingTool(BaseTool):
    """Tool for handling job-candidate matching operations."""
    
    name = "matching"
    description = "Handle job-candidate matching operations"
    # Define fields that will be set in __init__
    llm: ChatOpenAI = Field(default=None)
    vector_store: VectorStoreTool = Field(default=None)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self, vector_store=None):
        """Initialize the matching tool.
        
        Args:
            vector_store: Optional VectorStoreTool instance to use. If not provided, a new one will be created.
        """
        super().__init__()
        
        # Set up LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.3
        )
        
        # Initialize vector store
        self.vector_store = vector_store or VectorStoreTool()

    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Run matching operations."""
        try:
            if operation == "match_job_candidates":
                return self._match_job_candidates(**kwargs)
            elif operation == "match_candidate_jobs":
                return self._match_candidate_jobs(**kwargs)
            elif operation == "analyze_match":
                return self._analyze_match(**kwargs)
            else:
                return {
                    "status": "error",
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Async version of matching operations."""
        return self._run(operation, **kwargs)

    def _match_job_candidates(
        self,
        job_id: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> Dict[str, Any]:
        """Find matching candidates for a job."""
        try:
            # Get job details
            job_result = self.vector_store._run(
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
            
            # Find matching candidates
            candidates_result = self.vector_store._run(
                "search_candidates",
                query=job_data["content"],
                top_k=top_k
            )
            
            if candidates_result["status"] != "success":
                return candidates_result
            
            # Score and filter matches
            matches = []
            for candidate in candidates_result["results"]:
                # Analyze match using LLM
                analysis = self._analyze_match(
                    job_data["content"],
                    candidate["content"]
                )
                
                if analysis["score"] >= min_score:
                    matches.append({
                        "candidate_id": candidate["id"],
                        "candidate_name": candidate["name"],
                        "score": analysis["score"],
                        "analysis": analysis["details"]
                    })
            
            # Sort by score
            matches.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "status": "success",
                "job_id": job_id,
                "matches": matches
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _match_candidate_jobs(
        self,
        candidate_id: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> Dict[str, Any]:
        """Find matching jobs for a candidate."""
        try:
            # Get candidate details
            candidate_result = self.vector_store._run(
                "search_candidates",
                query=f"candidate_id:{candidate_id}",
                top_k=1
            )
            
            if candidate_result["status"] != "success" or not candidate_result["results"]:
                return {
                    "status": "error",
                    "error": "Candidate not found"
                }
            
            candidate_data = candidate_result["results"][0]
            
            # Find matching jobs
            jobs_result = self.vector_store._run(
                "search_jobs",
                query=candidate_data["content"],
                top_k=top_k
            )
            
            if jobs_result["status"] != "success":
                return jobs_result
            
            # Score and filter matches
            matches = []
            for job in jobs_result["results"]:
                # Analyze match using LLM
                analysis = self._analyze_match(
                    job["content"],
                    candidate_data["content"]
                )
                
                if analysis["score"] >= min_score:
                    matches.append({
                        "job_id": job["id"],
                        "job_title": job["title"],
                        "company": job["company"],
                        "score": analysis["score"],
                        "analysis": analysis["details"]
                    })
            
            # Sort by score
            matches.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "status": "success",
                "candidate_id": candidate_id,
                "matches": matches
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _analyze_match(
        self,
        job_content: str,
        candidate_content: str
    ) -> Dict[str, Any]:
        """Analyze the match between a job and candidate."""
        try:
            # Use LLM to analyze the match
            prompt = f"""Analyze the match between this job and candidate:
            
            Job:
            {job_content}
            
            Candidate:
            {candidate_content}
            
            Provide a detailed analysis including:
            1. Overall match score (0-1)
            2. Key matching skills and experience
            3. Areas of strength
            4. Potential gaps
            5. Recommendations
            
            Return the analysis in JSON format."""
            
            response = self.llm.invoke(prompt)
            
            # Parse the response
            analysis = parse_llm_json_response(response.content)
            
            return {
                "status": "success",
                "score": analysis.get("score", 0),
                "details": analysis
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 