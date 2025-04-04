from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.openai_service import OpenAIService
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from supabase._async.client import AsyncClient
from pinecone import Pinecone, Index
from app.config.settings import get_table_name

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
        
        # Pinecone Initialization
        self.pinecone = Pinecone(api_key=settings.pinecone_api_key, environment=settings.pinecone_environment)
        self.candidates_index_name = settings.pinecone_candidates_index
        self.jobs_index_name = settings.pinecone_jobs_index
        
        # Ensure indices exist
        # self._ensure_pinecone_index(self.candidates_index_name)
        # self._ensure_pinecone_index(self.jobs_index_name)

        # Use get_table_name for table names
        self.candidates_table = get_table_name("candidates")
        self.jobs_table = get_table_name("jobs")
        
        self.candidates_index: Index = self.pinecone.Index(self.candidates_index_name)
        self.jobs_index: Index = self.pinecone.Index(self.jobs_index_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI's API."""
        try:
            return await self.openai.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise Exception(f"Error generating embedding: {str(e)}")

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
            
            # Update candidate with embedding in dedicated column
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
    async def upsert_job(self, job_id: str, job_data: Dict[str, Any]) -> str:
        """
        Generate embeddings for job profile and store in Supabase.
        """
        try:
            logger.info(f"Generating embedding for job {job_id}")
            
            # Prepare text for embedding
            text_parts = []
            
            # Add title and company
            if job_data.get('job_title'):
                text_parts.append(f"Title: {job_data['job_title']}")
            if job_data.get('company_name'):
                text_parts.append(f"Company: {job_data['company_name']}")
            
            # Add key job details
            if job_data.get('key_responsibilities'):
                text_parts.append(f"Responsibilities: {', '.join(job_data['key_responsibilities'])}")
            if job_data.get('skills_must_have'):
                text_parts.append(f"Required Skills: {', '.join(job_data['skills_must_have'])}")
            if job_data.get('skills_preferred'):
                text_parts.append(f"Preferred Skills: {', '.join(job_data['skills_preferred'])}")
            
            # Add company details
            if job_data.get('company_mission'):
                text_parts.append(f"Company Mission: {job_data['company_mission']}")
            if job_data.get('company_vision'):
                text_parts.append(f"Company Vision: {job_data['company_vision']}")
            if job_data.get('company_culture'):
                text_parts.append(f"Company Culture: {job_data['company_culture']}")
            
            # Add role details
            if job_data.get('role_category'):
                text_parts.append(f"Role Categories: {', '.join(job_data['role_category'])}")
            if job_data.get('seniority'):
                text_parts.append(f"Seniority: {job_data['seniority']}")
            if job_data.get('scope_of_impact'):
                text_parts.append(f"Scope of Impact: {', '.join(job_data['scope_of_impact'])}")
            
            # Add technical details
            if job_data.get('tech_stack_must_haves'):
                text_parts.append(f"Required Tech Stack: {', '.join(job_data['tech_stack_must_haves'])}")
            if job_data.get('tech_stack_nice_to_haves'):
                text_parts.append(f"Preferred Tech Stack: {', '.join(job_data['tech_stack_nice_to_haves'])}")
            
            # Generate embedding from combined text
            text = " | ".join(text_parts)
            vector = await self.openai.generate_embedding(text)
            
            # Check if job exists
            response = await self.supabase.table(self.jobs_table).select("*").eq("id", job_id).execute()
            
            if not response.data:
                # Create new job
                insert_data = {
                    **job_data,
                    "embedding": vector,
                    "embedding_metadata": self._flatten_metadata(job_data),
                }
                insert_data_filtered = {k: v for k, v in insert_data.items() if v is not None}
                await self.supabase.table(self.jobs_table).insert(insert_data_filtered).execute()
            else:
                # Update existing job
                # Exclude 'id' from the update payload as it's GENERATED ALWAYS
                update_payload = {k: v for k, v in job_data.items() if k != 'id'} # Create payload without id
                update_data = {
                    **update_payload, # Spread the filtered job data
                    "embedding": vector,
                    "embedding_metadata": self._flatten_metadata(job_data), # Metadata can still use original job_data
                }
                # Filter out None values to avoid overwriting existing fields with NULL
                update_data_filtered = {k: v for k, v in update_data.items() if v is not None}
                await self.supabase.table(self.jobs_table).update(update_data_filtered).eq("id", job_id).execute()
            
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
                    "job_title": item.get("job_title"),
                    "company_name": item.get("company_name"),
                    "key_responsibilities": item.get("key_responsibilities", []),
                    "skills_must_have": item.get("skills_must_have", []),
                    "skills_preferred": item.get("skills_preferred", []),
                    "tech_stack_must_haves": item.get("tech_stack_must_haves", []),
                    "tech_stack_nice_to_haves": item.get("tech_stack_nice_to_haves", []),
                    "role_category": item.get("role_category", []),
                    "seniority": item.get("seniority"),
                    "scope_of_impact": item.get("scope_of_impact", []),
                    "company_mission": item.get("company_mission"),
                    "company_vision": item.get("company_vision"),
                    "company_culture": item.get("company_culture"),
                    "similarity": item.get("similarity", 0)
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