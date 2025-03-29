from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from app.config import get_settings, get_pinecone
from app.services.openai_service import OpenAIService
import os
import pinecone
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

settings = get_settings()
pinecone = get_pinecone()
openai = OpenAIService()

class PineconeService:
    def __init__(self):
        self.pinecone = pinecone
        self.jobs_index = pinecone.Index(settings.pinecone_jobs_index)
        self.candidates_index = pinecone.Index(settings.pinecone_candidates_index)
        self.openai = openai

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upsert_candidate(self, candidate_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Upsert candidate profile to Pinecone.
        """
        try:
            logger.info(f"Generating embedding for candidate {candidate_id}")
            # Convert profile data to text and generate embedding
            text = self.openai._prepare_text_for_embedding(profile_data)
            vector = await self.openai.generate_embedding(text)
            
            # Create metadata for the vector
            metadata = {
                'type': 'candidate',
                'candidate_id': candidate_id,
                **self._flatten_metadata(profile_data)
            }

            logger.info(f"Upserting candidate {candidate_id} to Pinecone")
            # Upsert to Pinecone
            self.candidates_index.upsert(
                vectors=[(candidate_id, vector, metadata)],
                namespace='candidates'
            )

            return candidate_id

        except Exception as e:
            logger.error(f"Error upserting candidate {candidate_id} to Pinecone: {str(e)}")
            raise Exception(f"Error upserting candidate to Pinecone: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upsert_job(self, job_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Upsert job profile to Pinecone.
        """
        try:
            logger.info(f"Generating embedding for job {job_id}")
            # Convert profile data to text and generate embedding
            text = self.openai._prepare_text_for_embedding(profile_data)
            vector = await self.openai.generate_embedding(text)
            
            # Create metadata for the vector
            metadata = {
                'type': 'job',
                'job_id': job_id,
                **self._flatten_metadata(profile_data)
            }

            logger.info(f"Upserting job {job_id} to Pinecone")
            # Upsert to Pinecone
            self.jobs_index.upsert(
                vectors=[(job_id, vector, metadata)],
                namespace='jobs'
            )

            return job_id

        except Exception as e:
            logger.error(f"Error upserting job {job_id} to Pinecone: {str(e)}")
            raise Exception(f"Error upserting job to Pinecone: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_matches(self, query_vector: List[float], top_k: int = 10, namespace: str = 'jobs') -> List[Dict[str, Any]]:
        """
        Query Pinecone for matches.
        """
        try:
            logger.info(f"Querying Pinecone for matches in namespace {namespace}")
            results = self.jobs_index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            return results.matches

        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            raise Exception(f"Error querying Pinecone: {str(e)}")

    def _flatten_metadata(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested dictionary for Pinecone metadata.
        """
        flattened = {}
        for key, value in data.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_metadata(value, new_key))
            elif isinstance(value, list):
                flattened[new_key] = ','.join(map(str, value))
            else:
                flattened[new_key] = str(value)
                
        return flattened 