from typing import Any, Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ..tools.vector_store import VectorStoreTool
from ..tools.matching import MatchingTool
from ..tools.communication import EmailTool
import logging

logger = logging.getLogger(__name__)

class JobMatchingChain:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        vector_store: Optional[VectorStoreTool] = None
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize tools
        if vector_store:
            logger.info("JobMatchingChain using provided VectorStoreTool instance")
            self.vector_store = vector_store
        else:
            logger.warning("⚠️ JobMatchingChain creating new VectorStoreTool - this should be avoided!")
            self.vector_store = VectorStoreTool()
            
        self.matching_tool = MatchingTool(vector_store=self.vector_store)
        self.email_tool = EmailTool()
        
        # Initialize chains
        self._initialize_chains()

    def _initialize_chains(self):
        """Initialize the matching chains."""
        # Job analysis chain
        job_analysis_prompt = PromptTemplate(
            input_variables=["job_data"],
            template="""Analyze this job posting and provide a detailed assessment:
            
            Job Data:
            {job_data}
            
            Provide analysis in JSON format with:
            1. Role requirements
            2. Key responsibilities
            3. Required skills
            4. Experience level
            5. Company culture
            6. Growth opportunities"""
        )
        
        self.job_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=job_analysis_prompt
        )
        
        # Match analysis chain
        match_analysis_prompt = PromptTemplate(
            input_variables=["job_analysis", "candidate_analysis", "match_score"],
            template="""Analyze the match between this job and candidate:
            
            Job Analysis:
            {job_analysis}
            
            Candidate Analysis:
            {candidate_analysis}
            
            Match Score:
            {match_score}
            
            Provide detailed analysis in JSON format with:
            1. Overall match assessment
            2. Key matching points
            3. Potential challenges
            4. Growth opportunities
            5. Recommendations for next steps"""
        )
        
        self.match_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=match_analysis_prompt
        )

    async def find_matches(
        self,
        job_id: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> Dict[str, Any]:
        """Find matching candidates for a job."""
        try:
            # Step 1: Get job details
            job_result = await self.vector_store._arun(
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
            
            # Step 2: Analyze job
            job_analysis = await self.job_analysis_chain.arun(
                job_data=job_data["content"]
            )
            
            # Step 3: Find matching candidates
            matches_result = await self.matching_tool._arun(
                "match_job_candidates",
                job_id=job_id,
                top_k=top_k,
                min_score=min_score
            )
            
            if matches_result["status"] != "success":
                return matches_result
            
            # Step 4: Analyze each match
            detailed_matches = []
            for match in matches_result["matches"]:
                # Get candidate details
                candidate_result = await self.vector_store._arun(
                    "search_candidates",
                    query=f"candidate_id:{match['candidate_id']}",
                    top_k=1
                )
                
                if candidate_result["status"] != "success" or not candidate_result["results"]:
                    continue
                
                candidate_data = candidate_result["results"][0]
                
                # Analyze match
                match_analysis = await self.match_analysis_chain.arun(
                    job_analysis=job_analysis,
                    candidate_analysis=candidate_data["content"],
                    match_score=match["score"]
                )
                
                detailed_matches.append({
                    "candidate_id": match["candidate_id"],
                    "candidate_name": match["candidate_name"],
                    "score": match["score"],
                    "analysis": match_analysis
                })
            
            # Step 5: Sort matches by score
            detailed_matches.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "status": "success",
                "job_id": job_id,
                "job_title": job_data["title"],
                "company": job_data["company"],
                "matches": detailed_matches
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
                # Generate personalized email
                email_content = await self.email_tool._arun(
                    "generate_email",
                    template="""Dear {candidate_name},

We're excited to inform you that your profile matches our {job_title} position at {company}.

Match Details:
- Match Score: {score}
- Key Matching Points: {matching_points}
- Next Steps: {next_steps}

We would love to discuss this opportunity with you. Please let us know if you're interested in moving forward.

Best regards,
The Recruitment Team""",
                    context={
                        "candidate_name": match["candidate_name"],
                        "job_title": match.get("job_title", "position"),
                        "company": match.get("company", "our company"),
                        "score": match["score"],
                        "matching_points": match["analysis"].get("key_matching_points", []),
                        "next_steps": match["analysis"].get("recommendations", [])
                    }
                )
                
                if email_content["status"] != "success":
                    continue
                
                # Send email
                email_result = await self.email_tool._arun(
                    "send_email",
                    to_email=match.get("candidate_email"),
                    subject=f"Job Match: {match.get('job_title', 'Position')} at {match.get('company', 'Our Company')}",
                    content=email_content["body"]
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