from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.openai_service import OpenAIService
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

settings = get_settings()
supabase = get_supabase_client()
openai = OpenAIService()

class VectorService:
    """
    Service for vector operations using Supabase with pgvector.
    Handles embedding generation, storage, and similarity search.
    """
    def __init__(self):
        self.supabase = supabase
        self.openai = openai
        self.candidates_table = "candidates_dev"
        self.jobs_table = "jobs_dev"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upsert_candidate(self, candidate_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Generate embeddings for candidate profile and store in Supabase.
        """
        try:
            logger.info(f"Generating embedding for candidate {candidate_id}")
            # Convert profile data to text and generate embedding
            text = self.openai._prepare_text_for_embedding(profile_data)
            vector = await self.openai.generate_embedding(text)
            
            # Check if candidate exists
            response = await self.supabase.table(self.candidates_table).select("*").eq("id", candidate_id).execute()
            
            if not response.data:
                logger.error(f"Candidate {candidate_id} not found in Supabase")
                raise ValueError(f"Candidate {candidate_id} not found")
            
            # Update candidate with embedding
            await self.supabase.table(self.candidates_table).update({
                "embedding": vector,
                "embedding_metadata": self._flatten_metadata(profile_data),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", candidate_id).execute()
            
            logger.info(f"Successfully stored embedding for candidate {candidate_id}")
            return candidate_id

        except Exception as e:
            logger.error(f"Error storing candidate embedding: {str(e)}")
            raise Exception(f"Error storing candidate embedding: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upsert_job(self, job_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Generate embeddings for job profile and store in Supabase.
        """
        try:
            logger.info(f"Generating embedding for job {job_id}")
            # Convert profile data to text and generate embedding
            text = self.openai._prepare_text_for_embedding(profile_data)
            vector = await self.openai.generate_embedding(text)
            
            # Check if job exists
            response = await self.supabase.table(self.jobs_table).select("*").eq("id", job_id).execute()
            
            if not response.data:
                # Create new job
                await self.supabase.table(self.jobs_table).insert({
                    "id": job_id,
                    "title": profile_data.get("title", ""),
                    "company": profile_data.get("company", ""),
                    "description": profile_data.get("description", ""),
                    "profile_json": profile_data,
                    "embedding": vector,
                    "embedding_metadata": self._flatten_metadata(profile_data),
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }).execute()
            else:
                # Update existing job
                await self.supabase.table(self.jobs_table).update({
                    "title": profile_data.get("title", ""),
                    "company": profile_data.get("company", ""),
                    "description": profile_data.get("description", ""),
                    "profile_json": profile_data,
                    "embedding": vector,
                    "embedding_metadata": self._flatten_metadata(profile_data),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", job_id).execute()
            
            logger.info(f"Successfully stored embedding for job {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Error storing job embedding: {str(e)}")
            raise Exception(f"Error storing job embedding: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_candidates(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query candidates based on vector similarity using pgvector.
        """
        try:
            logger.info(f"Querying candidates for semantic similarity")
            # Use pgvector's <-> operator for cosine distance (lower is more similar)
            # Note: The raw SQL is needed to use pgvector functionality
            rpc_response = await self.supabase.rpc(
                "match_candidates", 
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.5,  # Adjust as needed
                    "match_count": top_k
                }
            ).execute()
            
            if not rpc_response.data:
                return []
            
            # Format results
            results = []
            for item in rpc_response.data:
                results.append({
                    "id": item.get("id"),
                    "full_name": item.get("full_name"),
                    "email": item.get("email"),
                    "phone": item.get("phone"),
                    "similarity": item.get("similarity", 0),
                    "profile_json": item.get("profile_json", {})
                })
            
            return results

        except Exception as e:
            logger.error(f"Error querying candidates: {str(e)}")
            raise Exception(f"Error querying candidates: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_jobs(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query jobs based on vector similarity using pgvector.
        """
        try:
            logger.info(f"Querying jobs for semantic similarity")
            # Use pgvector's <-> operator for cosine distance (lower is more similar)
            rpc_response = await self.supabase.rpc(
                "match_jobs", 
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.5,  # Adjust as needed
                    "match_count": top_k
                }
            ).execute()
            
            if not rpc_response.data:
                return []
            
            # Format results
            results = []
            for item in rpc_response.data:
                results.append({
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "company": item.get("company"),
                    "description": item.get("description"),
                    "similarity": item.get("similarity", 0),
                    "profile_json": item.get("profile_json", {})
                })
            
            return results

        except Exception as e:
            logger.error(f"Error querying jobs: {str(e)}")
            raise Exception(f"Error querying jobs: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_and_store_candidate_embedding(self, candidate_id: str) -> Dict[str, Any]:
        """
        Generate embedding for a candidate and store it in Supabase.
        This is used when candidate data is already in Supabase.
        """
        try:
            # Get candidate data
            response = await self.supabase.table(self.candidates_table).select("*").eq("id", candidate_id).execute()
            if not response.data:
                raise ValueError(f"Candidate {candidate_id} not found")
            
            candidate = response.data[0]
            profile_data = candidate.get("profile_json", {})
            
            # Generate embedding
            text = self.openai._prepare_text_for_embedding(profile_data)
            vector = await self.openai.generate_embedding(text)
            
            # Update candidate with embedding
            update_response = await self.supabase.table(self.candidates_table).update({
                "embedding": vector,
                "embedding_metadata": self._flatten_metadata(profile_data),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", candidate_id).execute()
            
            if not update_response.data:
                raise Exception(f"Failed to update candidate {candidate_id} with embedding")
            
            logger.info(f"Successfully generated and stored embedding for candidate {candidate_id}")
            return update_response.data[0]
            
        except Exception as e:
            logger.error(f"Error generating embedding for candidate {candidate_id}: {str(e)}")
            raise Exception(f"Error generating embedding for candidate: {str(e)}")

    def _flatten_metadata(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested dictionary for metadata storage.
        """
        flattened = {}
        for key, value in data.items():
            # Skip large fields and non-indexable content
            if key in ['resume', 'raw_transcript', 'content']:
                continue
                
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_metadata(value, new_key))
            elif isinstance(value, list):
                try:
                    flattened[new_key] = ','.join(map(str, value))
                except:
                    # Skip if list can't be joined
                    pass
            else:
                # Store as string, limit length
                try:
                    str_value = str(value)
                    flattened[new_key] = str_value[:500] if len(str_value) > 500 else str_value
                except:
                    # Skip if can't be converted to string
                    pass
                
        return flattened 