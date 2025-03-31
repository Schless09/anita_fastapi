from typing import Dict, Any, Optional, List
import logging
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

class MatchingService:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.vector_service = VectorService()
    
    async def match_candidate_to_jobs(self, candidate_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Match a candidate to potential jobs based on their profile."""
        try:
            # 1. Generate embedding for candidate text
            candidate_text = self._prepare_candidate_for_matching(candidate_data)
            embedding = await self.openai_service.generate_embedding(candidate_text)
            
            # 2. Query vector database for matching jobs
            matches = await self.vector_service.query_jobs(embedding, top_k)
            
            # 3. Format and return results
            return [
                {
                    "job_id": job.get("id"),
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "similarity": job.get("similarity", 0),
                    "job_data": job.get("profile_json", {})
                }
                for job in matches
            ]
        
        except Exception as e:
            logger.error(f"Error matching candidate to jobs: {str(e)}")
            raise
    
    async def match_job_to_candidates(self, job_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Match a job to potential candidates based on its description."""
        try:
            # 1. Generate embedding for job text
            job_text = self._prepare_job_for_matching(job_data)
            embedding = await self.openai_service.generate_embedding(job_text)
            
            # 2. Query vector database for matching candidates
            matches = await self.vector_service.query_candidates(embedding, top_k)
            
            # 3. Format and return results
            return [
                {
                    "candidate_id": candidate.get("id"),
                    "name": candidate.get("full_name"),
                    "email": candidate.get("email"),
                    "similarity": candidate.get("similarity", 0),
                    "candidate_data": candidate.get("profile_json", {})
                }
                for candidate in matches
            ]
        
        except Exception as e:
            logger.error(f"Error matching job to candidates: {str(e)}")
            raise
    
    def _prepare_candidate_for_matching(self, candidate_data: Dict[str, Any]) -> str:
        """Format candidate data for matching."""
        sections = []
        profile = candidate_data.get("profile_json", {})
        
        # Add basic information
        sections.append(f"Name: {profile.get('full_name', '')}")
        sections.append(f"Current Role: {profile.get('current_role', '')}")
        sections.append(f"Current Company: {profile.get('current_company', '')}")
        
        # Add skills
        skills = profile.get("skills", [])
        if skills:
            sections.append(f"Skills: {', '.join(skills)}")
        
        # Add experience
        experience = profile.get("experience", [])
        if experience:
            sections.append("Experience:")
            for job in experience:
                sections.append(f"- {job.get('title')} at {job.get('company')} ({job.get('duration', '')})")
                sections.append(f"  {job.get('description', '')}")
        
        # Add education
        education = profile.get("education", [])
        if education:
            sections.append("Education:")
            for edu in education:
                sections.append(f"- {edu.get('degree')} at {edu.get('institution')} ({edu.get('year', '')})")
        
        return "\n".join(sections)
    
    def _prepare_job_for_matching(self, job_data: Dict[str, Any]) -> str:
        """Format job data for matching."""
        sections = []
        
        # Add basic information
        sections.append(f"Title: {job_data.get('title', '')}")
        sections.append(f"Company: {job_data.get('company', '')}")
        sections.append(f"Description: {job_data.get('description', '')}")
        
        # Add requirements
        requirements = job_data.get("requirements", [])
        if requirements:
            sections.append("Requirements:")
            for req in requirements:
                sections.append(f"- {req}")
        
        # Add skills
        skills = job_data.get("skills", [])
        if skills:
            sections.append(f"Skills: {', '.join(skills)}")
        
        # Add location and type
        sections.append(f"Location: {job_data.get('location', '')}")
        sections.append(f"Job Type: {job_data.get('job_type', '')}")
        
        return "\n".join(sections) 