from typing import Dict, Any, Optional, List
import logging
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.config.supabase import get_supabase_client

logger = logging.getLogger(__name__)
supabase = get_supabase_client()

class MatchingService:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.vector_service = VectorService()
        self.supabase = supabase
    
    async def match_candidate_to_jobs(self, candidate_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Match a candidate to potential jobs based on their stored embedding."""
        try:
            # 1. Fetch candidate's embedding from Supabase
            response = await self.supabase.table('candidates_dev').select('embedding').eq('id', candidate_id).maybe_single().execute()
            
            if not response.data or not response.data.get('embedding'):
                logger.warning(f"No embedding found for candidate {candidate_id}. Cannot perform matching.")
                return []
            
            embedding = response.data['embedding']
            
            # 2. Query vector database for matching jobs using the fetched embedding
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
            logger.error(f"Error matching candidate {candidate_id} to jobs: {str(e)}")
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
    
    def _prepare_job_for_matching(self, job_data: Dict[str, Any]) -> str:
        """Format job data for matching."""
        sections = []
        
        # Add basic information
        sections.append(f"Title: {job_data.get('title', '')}")
        sections.append(f"Company: {job_data.get('company', '')}")
        profile_json = job_data.get('profile_json', {})
        sections.append(f"Description: {profile_json.get('job_description', '')}")
        
        # Add requirements
        requirements = profile_json.get("requirements", [])
        if requirements:
            sections.append("Requirements:")
            for req in requirements:
                sections.append(f"- {req}")
        
        # Add skills
        skills = profile_json.get("skills", [])
        if skills:
            sections.append(f"Skills: {', '.join(skills)}")
        
        # Add location and type
        sections.append(f"Location: {profile_json.get('location', '')}")
        sections.append(f"Job Type: {profile_json.get('job_type', '')}")
        
        return "\n".join(sections) 