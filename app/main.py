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
import logging.handlers
from enum import Enum
import openai
import traceback
import json
from contextlib import asynccontextmanager

from app.config import (
    get_settings,
    get_openai_client,
    get_embeddings,
    get_pinecone,
    get_supabase_client,
    get_sendgrid_client
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
from app.api.webhook import handler as webhook_handler
from app.services.retell_service import RetellService
from app.services.openai_service import OpenAIService
from app.services.pinecone_service import PineconeService
from app.services.matching_service import MatchingService

# Configure logging for serverless environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,  # Force reconfiguration of the root logger
    handlers=[
        logging.StreamHandler()  # Console handler for terminal output
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure third-party loggers don't overwhelm our logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('pinecone').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)

# Test logging configuration
logger.info("\n=== Starting Anita AI Recruitment API ===")
logger.info(f"Environment: {os.getenv('VERCEL_ENV', 'development')}")
logger.info("=====================================\n")

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize core services first
logger.info("Initializing core services...")
try:
    settings = get_settings()
    logger.info("✅ Settings initialized successfully")
    
    supabase = get_supabase_client()
    logger.info("✅ Supabase client initialized successfully")
    
    # Initialize vector store and Pinecone
    logger.info("Initializing Pinecone and vector store...")
    pinecone_client = get_pinecone()
    logger.info("✅ Pinecone client initialized successfully")
    
    logger.info(f"Creating Pinecone indices with names: {settings.pinecone_jobs_index}, {settings.pinecone_candidates_index}")
    jobs_index = pinecone_client.Index(settings.pinecone_jobs_index)
    logger.info("✅ Jobs index initialized successfully")
    
    candidates_index = pinecone_client.Index(settings.pinecone_candidates_index)
    logger.info("✅ Candidates index initialized successfully")
    
    logger.info("Initializing VectorStoreTool...")
    vector_store = VectorStoreTool(jobs_index=jobs_index, candidates_index=candidates_index)
    logger.info("✅ VectorStoreTool initialized successfully")
    logger.info("⚠️ This is the ONLY VectorStoreTool instance that should be used throughout the application")
    
    # Initialize other services
    logger.info("Initializing remaining services...")
    retell = RetellService()
    logger.info("✅ RetellService initialized")
    
    openai_service = OpenAIService()
    logger.info("✅ OpenAIService initialized")
    
    pinecone = PineconeService()
    logger.info("✅ PineconeService initialized")
    
    matching = MatchingService()
    logger.info("✅ MatchingService initialized")
    
    candidate_service = CandidateService()
    logger.info("✅ CandidateService initialized")
    
    job_service = JobService()
    logger.info("✅ JobService initialized")
    
except Exception as e:
    logger.error(f"❌ Error during service initialization: {str(e)}")
    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    raise

# Initialize FastAPI app
logger.info("Initializing FastAPI application...")

# Remove legacy agent initialization since we're using the new LangChain agents
logger.info("Using LangChain agents for processing...")

# Replace deprecated on_event handlers with lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    This replaces the deprecated on_event("startup") and on_event("shutdown") handlers.
    """
    # Startup operations
    try:
        logger.info("Application startup complete")
        yield
    # Shutdown operations
    finally:
        logger.info("Application shutdown complete")

# Update app definition to use lifespan
app = FastAPI(
    title="Anita AI Recruitment API",
    description="API for AI-driven recruitment with enhanced candidate-job matching",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
llm = get_openai_client()
embeddings = get_embeddings()
pinecone = get_pinecone()
supabase = get_supabase_client()
sendgrid = get_sendgrid_client()

# Initialize tools
pdf_processor = PDFProcessor()
resume_parser = ResumeParser()
logger.info("🔄 Reusing shared VectorStoreTool instance for all agents and tools")
matching_tool = MatchingTool(vector_store=vector_store)
email_tool = EmailTool()

# Initialize chains with the shared vector_store
candidate_processing_chain = CandidateProcessingChain(vector_store=vector_store)
job_matching_chain = JobMatchingChain(vector_store=vector_store)
interview_scheduling_chain = InterviewSchedulingChain()
follow_up_chain = FollowUpChain(vector_store=vector_store)

# Initialize agents with emoji identifiers, passing the shared vector_store
candidate_intake_agent = CandidateIntakeAgent(vector_store=vector_store)
candidate_intake_agent.emoji = "👤"  # Person emoji for candidate intake
job_matching_agent = JobMatchingAgent(vector_store=vector_store)
job_matching_agent.emoji = "🔍"  # Magnifying glass for job matching
farming_matching_agent = FarmingMatchingAgent(vector_store=vector_store)
farming_matching_agent.emoji = "🌾"  # Wheat for farming
interview_agent = InterviewAgent(vector_store=vector_store)
interview_agent.emoji = "🎯"  # Target for interview
follow_up_agent = FollowUpAgent(vector_store=vector_store)
follow_up_agent.emoji = "📧"  # Envelope for follow-up

# Log agent initialization
logger.info("\n=== Initializing LangChain Agents ===")
logger.info(f"{candidate_intake_agent.emoji} Candidate Intake Agent")
logger.info(f"{job_matching_agent.emoji} Job Matching Agent")
logger.info(f"{farming_matching_agent.emoji} Farming Matching Agent")
logger.info(f"{interview_agent.emoji} Interview Agent")
logger.info(f"{follow_up_agent.emoji} Follow-up Agent")
logger.info("=====================================\n")

# Dependencies
def get_brain_agent():
    return BrainAgent(vector_store=vector_store)

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Retell webhook endpoint
@app.post("/webhook/retell")
async def retell_webhook(request: Request):
    """Handle Retell webhook events."""
    logger.info("🔍 FastAPI received Retell webhook request")
    logger.info(f"🔍 Request path: {request.url.path}")
    logger.info(f"🔍 Request method: {request.method}")
    logger.info(f"🔍 Request headers: {dict(request.headers)}")
    
    try:
        # Get raw body for signature verification
        body = await request.json()
        logger.info(f"🔍 Request body: {json.dumps(body, indent=2)}")
        
        response = await webhook_handler(request)
        logger.info(f"✅ Webhook handler returned response: {response}")
        
        # If status code is 204, return an empty response
        if response.get("statusCode") == 204:
            return Response(status_code=204)
            
        # Otherwise convert the webhook handler response to a FastAPI response
        status_code = response.get("statusCode", 500)
        headers = response.get("headers", {})
        body = json.loads(response.get("body", "{}"))
        
        return JSONResponse(
            status_code=status_code,
            content=body,
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ Error in FastAPI webhook route: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

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
logger.info("Using configured Pinecone instance...")

# Remove duplicate index initialization since we already have it from the config
logger.info(f"Using jobs index: {settings.pinecone_jobs_index}")
logger.info(f"Using candidates index: {settings.pinecone_candidates_index}")

# Candidate submission endpoint
@app.post("/candidates/", response_model=CandidateResponse)
async def create_candidate(
    background_tasks: BackgroundTasks,
    brain_agent: BrainAgent = Depends(get_brain_agent),
    name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    resume: UploadFile = File(...),
    linkedin: Optional[str] = Form(None)
):
    """Create a new candidate and process their profile."""
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
        
        # Create candidate data object
        candidate_data = {
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "linkedin": linkedin,
            "resume": resume,
            "resume_content": resume_content,
            "resume_filename": resume.filename
        }
        
        # Process candidate in background
        background_tasks.add_task(
            brain_agent.process_candidate,
            candidate=CandidateCreate(**candidate_data)
        )
        
        return {
            "id": "pending",  # Will be updated after processing
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "linkedin": linkedin,
            "status": "processing",
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
        
        matches = await brain_agent.match_candidates_to_job(
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
