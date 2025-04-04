from typing import Any, Dict, List, Optional, Literal, ClassVar, Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
import os
import logging
from pydantic import Field, PrivateAttr, BaseModel
from .base import parse_llm_json_response
from app.services.vector_service import VectorService
from app.config.settings import get_table_name
import traceback

logger = logging.getLogger(__name__)

class VectorStoreInput(BaseModel):
    """Input schema for vector store operations."""
    operation: Literal["store_job", "store_candidate", "search_jobs", "search_candidates"]
    job_data: Optional[Dict[str, Any]] = None
    candidate_data: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

class JobSearchInput(BaseModel):
    query: str = Field(description="Semantic query based on candidate preferences, skills, and experience.")
    # Add filters later if needed

class VectorStoreTool(BaseTool):
    """Tool for managing vector store operations. Implements singleton pattern."""
    
    name = "vector_store"
    description = "Manage vector store operations for jobs and candidates. Use for storing or searching job and candidate data."
    args_schema = VectorStoreInput
    
    # Class-level singleton instance
    _instance: ClassVar[Optional['VectorStoreTool']] = None
    
    # Private instance state using PrivateAttr
    _initialized: bool = PrivateAttr(default=False)
    
    # Define fields that will be set in __init__
    embeddings: OpenAIEmbeddings = Field(default=None)
    llm: ChatOpenAI = Field(default=None)
    vector_service: VectorService = Field(default=None)
    jobs_table: str = Field(default=None)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreTool, cls).__new__(cls)
            # Initialize PrivateAttr
            object.__setattr__(cls._instance, '_initialized', False)
        return cls._instance
    
    def __init__(self):
        """Initialize the vector store tool."""
        if not self._initialized:
            super().__init__()
            
            # Initialize Vector service
            self.vector_service = VectorService()
            
            # Create OpenAI embeddings
            self.embeddings = OpenAIEmbeddings()
            
            # Set up LLM
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                temperature=0.3
            )
            
            # Set initialized using PrivateAttr
            object.__setattr__(self, '_initialized', True)
            
            # Get dynamic table name
            self.jobs_table = get_table_name("jobs")
    
    async def _initialize_async(self):
        """Initialize async components."""
        try:
            # Test Supabase connection
            await self.vector_service.supabase.table("jobs_dev").select("count").execute()
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"Error initializing async components: {str(e)}")
            raise
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Run vector store operations."""
        try:
            if operation == "store_job":
                return self._store_job(**kwargs)
            elif operation == "store_candidate":
                return self._store_candidate(**kwargs)
            elif operation == "search_jobs":
                return self._search_jobs(**kwargs)
            elif operation == "search_candidates":
                return self._search_candidates(**kwargs)
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
        """Async version of vector store operations."""
        try:
            if operation == "store_job":
                return await self._store_job(**kwargs)
            elif operation == "store_candidate":
                return await self._store_candidate(**kwargs)
            elif operation == "search_jobs":
                return await self._search_jobs(**kwargs)
            elif operation == "search_candidates":
                return await self._search_candidates(**kwargs)
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

    async def _store_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a job in the vector store."""
        try:
            # Generate embedding and store in Supabase
            job_id = job_data.get("id")
            if not job_id:
                return {
                    "status": "error",
                    "error": "Job ID is required"
                }
            
            # Store the job
            await self.vector_service.upsert_job(job_id, job_data)
            
            return {
                "status": "success",
                "message": f"Job {job_id} stored successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _store_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a candidate in the vector store."""
        try:
            # Generate embedding and store in Supabase
            candidate_id = candidate_data.get("id")
            if not candidate_id:
                return {
                    "status": "error",
                    "error": "Candidate ID is required"
                }
            
            # Store the candidate
            await self.vector_service.upsert_candidate(candidate_id, candidate_data)
            
            return {
                "status": "success",
                "message": f"Candidate {candidate_id} stored successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _search_jobs(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for jobs using semantic similarity."""
        try:
            # Generate embedding for query
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Query jobs using embedding
            results = await self.vector_service.query_jobs(query_embedding, top_k)
            
            # Format results
            formatted_results = [
                {
                    "id": job.get("id"),
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "content": str(job.get("profile_json", {})),
                    "similarity": job.get("similarity", 0)
                }
                for job in results
            ]
            
            return {
                "status": "success",
                "results": formatted_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _search_candidates(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for candidates using semantic similarity."""
        try:
            # Generate embedding for query
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Query candidates using embedding
            results = await self.vector_service.query_candidates(query_embedding, top_k)
            
            # Format results
            formatted_results = [
                {
                    "id": candidate.get("id"),
                    "name": candidate.get("full_name"),
                    "email": candidate.get("email"),
                    "content": str(candidate.get("profile_json", {})),
                    "similarity": candidate.get("similarity", 0)
                }
                for candidate in results
            ]
            
            return {
                "status": "success",
                "results": formatted_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def store_candidate_vector(self, candidate_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Store candidate vector in Supabase."""
        try:
            # Update candidate with embedding
            await self.vector_service.upsert_candidate(candidate_id, metadata)
            logger.debug(f"Stored vector for candidate {candidate_id}")
        except Exception as e:
            logger.error(f"Error storing candidate vector: {e}")
            raise
    
    async def store_job_vector(self, job_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Store job vector in Supabase."""
        try:
            await self.vector_service.upsert_job(job_id, metadata)
            logger.debug(f"Stored vector for job {job_id}")
        except Exception as e:
            logger.error(f"Error storing job vector: {e}")
            raise
    
    async def query_candidates(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query candidates based on vector similarity."""
        try:
            return await self.vector_service.query_candidates(query_vector, top_k)
        except Exception as e:
            logger.error(f"Error querying candidates: {e}")
            raise
    
    async def query_jobs(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query jobs based on vector similarity."""
        try:
            return await self.vector_service.query_jobs(query_vector, top_k)
        except Exception as e:
            logger.error(f"Error querying jobs: {e}")
            raise

    async def search_jobs(self, query: str) -> List[Dict[str, Any]]:
        """Search for jobs using semantic similarity."""
        try:
            # Generate embedding for query
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Search Pinecone job index
            search_results = self.vector_service.jobs_index.query(
                vector=query_embedding,
                top_k=5, # Adjust top_k as needed
                include_metadata=True 
            )
            
            # Extract job IDs and scores
            matches = search_results.get('matches', [])
            job_ids = [match['id'] for match in matches]
            scores_map = {match['id']: match['score'] for match in matches}

            if not job_ids:
                return [{"message": "No matching jobs found in vector store."}]

            # Fetch full job details from Supabase using job_ids
            # Use the dynamic table name attribute
            response = await self.vector_service.supabase.table(self.jobs_table).select("*").in_('id', job_ids).execute()

            if response.data:
                # Combine Supabase data with Pinecone scores
                detailed_jobs = []
                for job in response.data:
                    job_id = job.get('id')
                    score = scores_map.get(job_id)
                    if score is not None:
                        job['match_score'] = score # Add score to the job data
                        detailed_jobs.append(job)
                # Sort by score descending
                detailed_jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
                logger.info(f"Found {len(detailed_jobs)} detailed jobs from vector search.")
                return detailed_jobs
            else:
                logger.warning(f"Vector search found job IDs {job_ids}, but failed to fetch details from Supabase.")
                return [{"message": "Found matches in vector store but failed to fetch details."}]

        except Exception as e:
            logger.error(f"Error during vector store job search: {e}")
            logger.error(traceback.format_exc())
            return [{"error": f"An internal error occurred during job search: {str(e)}"}] 