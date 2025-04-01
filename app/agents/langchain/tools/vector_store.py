from typing import Any, Dict, List, Optional, Literal, ClassVar
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
import os
import logging
from pydantic import Field, PrivateAttr, BaseModel
from .base import parse_llm_json_response
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

class VectorStoreInput(BaseModel):
    """Input schema for vector store operations."""
    operation: Literal["store_job", "store_candidate", "search_jobs", "search_candidates"]
    job_data: Optional[Dict[str, Any]] = None
    candidate_data: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

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
        return self._run(operation, **kwargs)

    def _store_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a job in the vector store."""
        try:
            # Generate embedding and store in Supabase
            job_id = job_data.get("id")
            if not job_id:
                return {
                    "status": "error",
                    "error": "Job ID is required"
                }
            
            # Since this is a synchronous method, we can't use async directly
            # We'll return a message that the job is being processed
            # The actual storage will happen asynchronously
            
            # Create a task to store the job
            import asyncio
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(self.vector_service.upsert_job(job_id, job_data))
            
            return {
                "status": "success",
                "message": f"Job {job_id} is being processed"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _store_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a candidate in the vector store."""
        try:
            # Generate embedding and store in Supabase
            candidate_id = candidate_data.get("id")
            if not candidate_id:
                return {
                    "status": "error",
                    "error": "Candidate ID is required"
                }
            
            # Since this is a synchronous method, we can't use async directly
            # We'll return a message that the candidate is being processed
            # The actual storage will happen asynchronously
            
            # Create a task to store the candidate
            import asyncio
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(self.vector_service.upsert_candidate(candidate_id, candidate_data))
            
            return {
                "status": "success",
                "message": f"Candidate {candidate_id} is being processed"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _search_jobs(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for jobs using semantic similarity."""
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Query jobs using embedding
            import asyncio
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self.vector_service.query_jobs(query_embedding, top_k))
            
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

    def _search_candidates(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for candidates using semantic similarity."""
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Query candidates using embedding
            import asyncio
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self.vector_service.query_candidates(query_embedding, top_k))
            
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