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
    get_supabase_client
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
        brain_agent_instance = BrainAgent(vector_store=vector_store)
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

# Dependencies
async def get_brain_agent():
    global brain_agent_instance
    if brain_agent_instance is None:
        raise HTTPException(
            status_code=500,
            detail="Brain agent not initialized"
        )
    return brain_agent_instance

async def get_vector_store():
    global vector_store
    if vector_store is None:
        raise HTTPException(
            status_code=500,
            detail="Vector store not initialized"
        )
    return vector_store

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

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Get Vercel protection bypass secret
VERCEL_PROTECTION_BYPASS = os.getenv('VERCEL_PROTECTION_BYPASS')
if not VERCEL_PROTECTION_BYPASS:
    raise ValueError("VERCEL_PROTECTION_BYPASS environment variable is not set")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

class RetellCallStatus(str, Enum):
    CREATED = "created"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    ERROR = "error"
    ERROR_UNKNOWN = "error_unknown"

class JobMatchRequest(BaseModel):
    job_id: str
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

# Candidate submission endpoint
@app.post("/candidates/", response_model=CandidateResponse)
async def create_candidate(
    background_tasks: BackgroundTasks,
    brain_agent: BrainAgent = Depends(get_brain_agent),
    name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    linkedin: Optional[str] = Form(None),
    resume: UploadFile = File(...)
):
    try:
        logger.info(f"👤 Processing new candidate submission: {email}")
        
        # Validate file is PDF
        if not resume.content_type == 'application/pdf':
            raise HTTPException(
                status_code=400,
                detail="Resume must be a PDF file"
            )
            
        # Read resume content
        resume_content = await resume.read()
        
        # Generate candidate ID upfront
        candidate_id = str(uuid.uuid4())
        
        # Create candidate service
        candidate_service = CandidateService()
        
        # Extract first name and last name
        first_name = name.split(' ')[0] if ' ' in name else name
        last_name = ' '.join(name.split(' ')[1:]) if ' ' in name else ''
        
        # Quick extract of current role/company from resume
        current_role = "current role"
        current_company = "current company"
        try:
            # Initialize OpenAI service
            openai_service = OpenAIService()
            
            # Extract text from PDF
            pdf_processor = PDFProcessor()
            pdf_result = await pdf_processor._arun(resume_content)
            if pdf_result["status"] == "success":
                text = pdf_result["text_content"]
                quick_info = await openai_service.quick_extract_current_position(text)
                current_role = quick_info.get('current_role', 'current role')
                current_company = quick_info.get('current_company', 'current company')
        except Exception as e:
            logger.warning(f"⚠️ Error in quick resume extraction: {str(e)}")
        
        # Prepare data for Supabase
        submission_data = {
            'id': candidate_id,
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'phone': phone_number,
            'linkedin_url': linkedin,
            'resume_content': resume_content,
            'resume_filename': resume.filename
        }
        
        # Store in Supabase
        stored_candidate = await candidate_service.process_candidate_submission(submission_data)
        logger.info(f"✅ Candidate {candidate_id} stored in Supabase")
        
        # Schedule Retell call immediately
        logger.info(f"📞 Scheduling Retell call for candidate {candidate_id}")
        retell_service = RetellService()
        call_result = await retell_service.schedule_call(
            candidate_id=candidate_id,
            dynamic_variables={
                'first_name': first_name,
                'email': email,
                'current_company': current_company,
                'current_title': current_role,
                'phone': phone_number
            }
        )
        logger.info(f"✅ Retell call scheduled for candidate {candidate_id}: {call_result.get('call_id', 'unknown')}")
        
        # Create candidate data object for background processing
        candidate_data = {
            "id": candidate_id,
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "linkedin": linkedin,
            "resume_content": resume_content,  # Binary PDF content
            "resume_filename": resume.filename
        }
        
        # Process candidate in background
        candidate_create = CandidateCreate(**candidate_data)
        background_tasks.add_task(
            brain_agent.handle_candidate_submission,
            candidate_data=candidate_create
        )
        
        return {
            "id": candidate_id,
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "linkedin": linkedin,
            "status": "processing",
            "call_id": call_result.get("call_id"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing candidate: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing candidate: {str(e)}"
        )

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
