from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import logging.config
import logging.handlers
from enum import Enum
import openai
import traceback
import json
from contextlib import asynccontextmanager
import uuid
import tempfile
import asyncio
from retell import Retell

from app.config import (
    get_settings,
    get_openai_client,
    get_embeddings,
    get_sendgrid_client,
    get_sendgrid_webhook_url,
    get_supabase_client,
)

from app.agents.langchain.agents.candidate_intake_agent import CandidateIntakeAgent
from app.agents.langchain.agents.job_matching_agent import JobMatchingAgent
from app.agents.langchain.agents.farming_matching_agent import FarmingMatchingAgent
from app.agents.langchain.agents.interview_agent import InterviewAgent
from app.agents.langchain.agents.follow_up_agent import FollowUpAgent

from app.agents.langchain.tools.document_processing import PDFProcessor, ResumeParser
from app.agents.langchain.tools.vector_store import VectorStoreTool
from app.agents.langchain.tools.matching import MatchingTool
from app.agents.langchain.tools.communication import EmailTool

from app.agents.langchain.chains.candidate_processing import CandidateProcessingChain
from app.agents.langchain.chains.job_matching import JobMatchingChain
from app.agents.langchain.chains.interview_scheduling import InterviewSchedulingChain
from app.agents.langchain.chains.follow_up import FollowUpChain

from app.services.candidate_service import CandidateService
from app.services.job_service import JobService
from app.agents.brain_agent import BrainAgent
from app.schemas.candidate import CandidateCreate, CandidateResponse, CandidateUpdate
from app.schemas.job import JobCreate, JobResponse, JobUpdate
from app.schemas.matching import MatchingResponse
from app.api.webhook import router as webhook_router
from app.services.retell_service import RetellService
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.services.matching_service import MatchingService
from app.routes.jobs import router as jobs_router
# Import the new webhook router
from app.routes.webhooks import router as webhook_jobs_router
from app.routes.candidates import router as candidates_router
from app.dependencies import get_brain_agent

# Load environment variables first
load_dotenv()

# Configure logging from config file
logging.config.fileConfig('app/config/logging.conf')
logger = logging.getLogger('app')

# Test logging configuration
logger.info("🚀 Starting Anita AI with debug logging enabled")
logger.debug("Debug logging is enabled")

# Suppress noisy logs from other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Create FastAPI app
app = FastAPI(
    title="Anita AI Recruitment API",
    description="API for AI-driven recruitment with enhanced candidate-job matching",
    version="2.0.0"
)

# Global service instances
vector_store = None
brain_agent_instance = None
core_services = {}
agents = {}

# Initialize services
settings = get_settings()
llm = get_openai_client()
embeddings = get_embeddings()
supabase = get_supabase_client()
sendgrid = get_sendgrid_client()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vector_store, brain_agent_instance
    try:
        logger.info("Initializing services...")
        
        # Initialize vector store
        from app.agents.langchain.tools.vector_store import VectorStoreTool
        vector_store = VectorStoreTool()
        await vector_store._initialize_async()
        
        # Initialize brain agent
        from app.agents.brain_agent import BrainAgent
        brain_agent_instance = BrainAgent()
        await brain_agent_instance._initialize_async()
        
        logger.info("✅ Services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error initializing services: {str(e)}")
        raise

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# Import routers
from app.api.webhook import router as webhook_router

# Include routers
app.include_router(webhook_router, prefix="/webhook", tags=["webhook"])
app.include_router(jobs_router, tags=["jobs"])
app.include_router(candidates_router, tags=["candidates"])
# Include the new webhook router
app.include_router(webhook_jobs_router, prefix="/api/v1", tags=["Webhooks"])

def get_service(service_name: str):
    return core_services.get(service_name)

def get_agent(agent_name: str):
    return agents.get(agent_name)

@app.get("/health")
async def health_check():
    """Health check endpoint with initialization status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "vector_store": vector_store is not None,
            "brain_agent": brain_agent_instance is not None,
            "core_services": list(core_services.keys()),
            "agents": list(agents.keys())
        }
    }

class RetellCallStatus(str, Enum):
    CREATED = "created"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    ERROR = "error"
    ERROR_UNKNOWN = "error_unknown"

class JobMatchRequest(BaseModel):
    top_k: Optional[int] = Field(default=5, gt=0, le=100)

class ScopeOfImpact(str, Enum):
    TEAM = "Team"
    DEPARTMENT = "Department"
    COMPANY = "Company"
    INDUSTRY = "Industry"

class CallStatusRequest(BaseModel):
    """Request model for checking call status"""
    candidate_id: str
    call_id: str

# Remove duplicate Pinecone initialization since we're using the config service
logger.info("Using Supabase for vector storage...")

# Job matching endpoint
@app.post("/jobs/{job_id}/match", response_model=MatchingResponse)
async def match_candidates(
    job_id: str,
    request: JobMatchRequest,
    brain_agent: BrainAgent = Depends(get_brain_agent)
):
    """Match candidates to a specific job."""
    try:
        logger.info(f"🔍 Matching candidates for job: {job_id}")
        
        matches = await brain_agent.handle_job_matching(
            job_id=job_id,
            top_k=request.top_k
        )
        
        return {
            "job_id": job_id,
            "matches": matches,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error matching candidates: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error matching candidates: {str(e)}"
        )

# Call status check endpoint
@app.post("/calls/status", response_model=Dict[str, Any])
async def check_call_status(
    request: CallStatusRequest,
    brain_agent: BrainAgent = Depends(get_brain_agent)
):
    """Check the status of a Retell call."""
    try:
        logger.info(f"📞 Checking call status for candidate: {request.candidate_id}")
        
        status = await brain_agent.check_call_status(
            candidate_id=request.candidate_id,
            call_id=request.call_id
        )
        
        return {
            "candidate_id": request.candidate_id,
            "call_id": request.call_id,
            "status": status,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking call status: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking call status: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}
