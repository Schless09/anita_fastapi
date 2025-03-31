from typing import Any, Dict, List, Optional, Literal, ClassVar
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
from pinecone import Pinecone as PineconeClient
import os
import logging
from pydantic import Field, PrivateAttr, BaseModel
from .base import parse_llm_json_response
from app.config import get_pinecone
from app.services.pinecone_service import PineconeService

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
    jobs_index: Any = Field(default=None)
    candidates_index: Any = Field(default=None)
    pinecone_client: Any = Field(default=None)
    llm: ChatOpenAI = Field(default=None)
    pinecone_service: PineconeService = Field(default=None)
    pinecone: Any = Field(default=None)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Creating new VectorStoreTool instance")
            cls._instance = super(VectorStoreTool, cls).__new__(cls)
            # Initialize PrivateAttr
            object.__setattr__(cls._instance, '_initialized', False)
        return cls._instance
    
    def __init__(self, jobs_index=None, candidates_index=None):
        """Initialize the vector store tool."""
        if not self._initialized:
            logger.info("Initializing VectorStoreTool")
            super().__init__()
            
            # Initialize Pinecone service
            self.pinecone = get_pinecone()
            self.pinecone_service = PineconeService()
            
            # Create OpenAI embeddings
            self.embeddings = OpenAIEmbeddings()
            
            # Set up LLM
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                temperature=0.3
            )
            
            # Use provided indices or initialize new ones
            if jobs_index and candidates_index:
                logger.info("Using provided vector indices.")
                self.pinecone_client = None
                # Create langchain Pinecone vector stores from raw indices
                self.jobs_index = Pinecone.from_existing_index(
                    index_name=os.getenv("PINECONE_JOBS_INDEX", "jobs"),
                    embedding=self.embeddings,
                    text_key="content"
                )
                self.candidates_index = Pinecone.from_existing_index(
                    index_name=os.getenv("PINECONE_CANDIDATES_INDEX", "candidates"),
                    embedding=self.embeddings,
                    text_key="content"
                )
            else:
                logger.info("Initializing new vector indices...")
                # Initialize Pinecone client
                self.pinecone_client = PineconeClient(
                    api_key=os.getenv("PINECONE_API_KEY")
                )
                
                # Get index names from environment
                jobs_index_name = os.getenv("PINECONE_JOBS_INDEX", "jobs")
                candidates_index_name = os.getenv("PINECONE_CANDIDATES_INDEX", "candidates")
                
                # Initialize langchain vector store indices
                self.jobs_index = Pinecone.from_existing_index(
                    index_name=jobs_index_name,
                    embedding=self.embeddings,
                    text_key="content"
                )
                self.candidates_index = Pinecone.from_existing_index(
                    index_name=candidates_index_name,
                    embedding=self.embeddings,
                    text_key="content"
                )
                logger.info(f"Initialized vector indices: {jobs_index_name}, {candidates_index_name}")
            
            # Set initialized using PrivateAttr
            object.__setattr__(self, '_initialized', True)
            logger.info("✅ VectorStoreTool initialization complete")
    
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
            # Create document
            doc = Document(
                page_content=str(job_data),
                metadata={
                    "type": "job",
                    "job_id": job_data.get("id"),
                    "title": job_data.get("title"),
                    "company": job_data.get("company")
                }
            )
            
            # Add to vector store
            self.jobs_index.add_documents([doc])
            
            return {
                "status": "success",
                "message": f"Job {job_data.get('id')} stored successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _store_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a candidate in the vector store."""
        try:
            # Create document
            doc = Document(
                page_content=str(candidate_data),
                metadata={
                    "type": "candidate",
                    "candidate_id": candidate_data.get("id"),
                    "name": candidate_data.get("name"),
                    "email": candidate_data.get("email")
                }
            )
            
            # Add to vector store
            self.candidates_index.add_documents([doc])
            
            return {
                "status": "success",
                "message": f"Candidate {candidate_data.get('id')} stored successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _search_jobs(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for jobs using semantic similarity."""
        try:
            # Perform similarity search
            results = self.jobs_index.similarity_search(
                query,
                k=top_k
            )
            
            # Format results
            formatted_results = [
                {
                    "id": doc.metadata.get("job_id"),
                    "title": doc.metadata.get("title"),
                    "company": doc.metadata.get("company"),
                    "content": doc.page_content
                }
                for doc in results
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
            # Perform similarity search
            results = self.candidates_index.similarity_search(
                query,
                k=top_k
            )
            
            # Format results
            formatted_results = [
                {
                    "id": doc.metadata.get("candidate_id"),
                    "name": doc.metadata.get("name"),
                    "email": doc.metadata.get("email"),
                    "content": doc.page_content
                }
                for doc in results
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
    
    def store_candidate_vector(self, candidate_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Store candidate vector in the vector store."""
        try:
            self.pinecone_service.store_candidate_vector(candidate_id, vector, metadata)
            logger.debug(f"Stored vector for candidate {candidate_id}")
        except Exception as e:
            logger.error(f"Error storing candidate vector: {e}")
            raise
    
    def store_job_vector(self, job_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Store job vector in the vector store."""
        try:
            self.pinecone_service.store_job_vector(job_id, vector, metadata)
            logger.debug(f"Stored vector for job {job_id}")
        except Exception as e:
            logger.error(f"Error storing job vector: {e}")
            raise
    
    def query_candidates(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query candidates based on vector similarity."""
        try:
            return self.pinecone_service.query_candidates(query_vector, top_k)
        except Exception as e:
            logger.error(f"Error querying candidates: {e}")
            raise
    
    def query_jobs(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query jobs based on vector similarity."""
        try:
            return self.pinecone_service.query_jobs(query_vector, top_k)
        except Exception as e:
            logger.error(f"Error querying jobs: {e}")
            raise 