from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.vector_service import VectorService
from app.services.openai_service import OpenAIService

# Initialize services
settings = get_settings()
supabase = get_supabase_client()
vector_service = VectorService()
openai = OpenAIService()

class JobService:
    def __init__(self):
        self.supabase = supabase
        self.vector_service = VectorService()
        self.openai = openai
        
    async def process_job_submission(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new job submission.
        Creates job in DB, adds vector embedding, and returns success status.
        """
        try:
            # 1. Validate job data
            if not job_data.get("title") or not job_data.get("company"):
                raise ValueError("Job title and company are required")
                
            # 2. Combine relevant data for embedding
            embedding_data = self._prepare_job_data_for_embedding(job_data)
            
            # 3. Send combined data to vector store
            logger.info(f"Storing job {job_data.get('id')} in vector store")
            vector_id = await self.vector_service.upsert_job(
                job_data.get("id"),
                embedding_data
            )
            
            # 4. Update job with vector ID
            job_data.update({
                'vector_id': vector_id,
                'updated_at': datetime.utcnow().isoformat()
            })
            
            # 5. Return success response
            return {
                "id": job_data.get("id"),
                "title": job_data.get("title"),
                "company": job_data.get("company"),
                "status": "processed",
                "vector_id": vector_id
            }
            
        except Exception as e:
            logger.error(f"Error processing job: {str(e)}")
            raise
    
    def _prepare_job_data_for_embedding(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine job and company data for vector store storage.
        """
        # Extract relevant fields
        embedding_data = {
            "id": job_data.get("id"),
            "title": job_data.get("title", ""),
            "company": job_data.get("company", ""),
            "description": job_data.get("description", ""),
            "requirements": job_data.get("requirements", []),
            "skills": job_data.get("skills", []),
            "location": job_data.get("location", ""),
            "salary_range": job_data.get("salary_range", ""),
            "job_type": job_data.get("job_type", ""),
            "industry": job_data.get("industry", ""),
            "experience_level": job_data.get("experience_level", "")
        }
        
        # Add additional data about company if available
        if company_data := job_data.get("company_data"):
            embedding_data["company_description"] = company_data.get("description", "")
            embedding_data["company_size"] = company_data.get("size", "")
            embedding_data["company_industry"] = company_data.get("industry", "")
            embedding_data["company_website"] = company_data.get("website", "")
            
        return embedding_data 