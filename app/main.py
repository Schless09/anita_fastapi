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
import uuid
import tempfile
import asyncio

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

# Global service instances
vector_store = None
brain_agent_instance = None
core_services = {}
agents = {}

async def initialize_core_services():
    """Initialize core services asynchronously."""
    logger.info("Initializing core services...")
    
    # Get or create the singleton VectorStoreTool instance
    vector_store = VectorStoreTool.get_instance()
    
    # Initialize other core services
    services = {
        'matching_tool': MatchingTool(vector_store=vector_store),
        'pdf_processor': PDFProcessor(),
        'resume_parser': ResumeParser(),
        'email_tool': EmailTool()
    }
    
    logger.info("✅ Core services initialized")
    return services

async def initialize_chains(vector_store, services):
    """Initialize chains asynchronously."""
    logger.info("Initializing chains...")
    
    chains = {
        'candidate_processing': CandidateProcessingChain(
            vector_store=vector_store
        ),
        'job_matching': JobMatchingChain(
            vector_store=vector_store
        ),
        'interview_scheduling': InterviewSchedulingChain(),
        'follow_up': FollowUpChain(vector_store=vector_store)
    }
    
    logger.info("✅ Chains initialized")
    return chains

async def initialize_agents(vector_store, services, chains):
    """Initialize agents asynchronously."""
    logger.info("\n=== Initializing LangChain Agents ===")
    
    # Create initialization tasks
    tasks = [
        asyncio.create_task(initialize_agent(
            "candidate_intake",
            CandidateIntakeAgent,
            vector_store,
            services,
            chains,
            "👤"
        )),
        asyncio.create_task(initialize_agent(
            "job_matching",
            JobMatchingAgent,
            vector_store,
            services,
            chains,
            "🔍"
        )),
        asyncio.create_task(initialize_agent(
            "farming_matching",
            FarmingMatchingAgent,
            vector_store,
            services,
            chains,
            "🌾"
        )),
        asyncio.create_task(initialize_agent(
            "interview",
            InterviewAgent,
            vector_store,
            services,
            chains,
            "🎯"
        )),
        asyncio.create_task(initialize_agent(
            "follow_up",
            FollowUpAgent,
            vector_store,
            services,
            chains,
            "📧"
        ))
    ]
    
    # Wait for all agents to initialize
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check for any initialization errors
    agents = {}
    for name, result in zip(["candidate_intake", "job_matching", "farming_matching", "interview", "follow_up"], results):
        if isinstance(result, Exception):
            logger.error(f"Failed to initialize {name} agent: {result}")
        else:
            agents[name] = result
    
    logger.info("=====================================\n")
    return agents

async def initialize_agent(name, agent_class, vector_store, services, chains, emoji):
    """Initialize a single agent asynchronously."""
    try:
        logger.info(f"Initializing {emoji} {name} agent...")
        
        # Only pass vector_store - each agent handles its own tool creation internally
        agent = agent_class(vector_store=vector_store)
        agent.emoji = emoji
        logger.info(f"✅ {emoji} {name} agent initialized")
        return agent
    except Exception as e:
        logger.error(f"Error initializing {name} agent: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Initialize services once and reuse them throughout the application lifecycle.
    """
    global core_services, agents, brain_agent_instance, vector_store
    
    # Startup
    try:
        logger.info("Initializing core services...")
        
        # Initialize vector store first (singleton)
        vector_store = VectorStoreTool.get_instance()
        logger.info("✅ VectorStoreTool instance initialized (singleton)")
        
        # Initialize services, chains, and agents asynchronously
        core_services = await initialize_core_services()
        chains = await initialize_chains(vector_store, core_services)
        agents = await initialize_agents(vector_store, core_services, chains)
        
        # Initialize brain agent once
        if brain_agent_instance is None:
            logger.info("Initializing BrainAgent...")
            brain_agent_instance = BrainAgent(
                vector_store=vector_store,
                candidate_intake_agent=agents.get('candidate_intake'),
                job_matching_agent=agents.get('job_matching'),
                interview_agent=agents.get('interview'),
                follow_up_agent=agents.get('follow_up')
            )
            logger.info("✅ BrainAgent initialized")
        
        logger.info("Application startup complete")
        yield
        
    # Shutdown
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

# Dependencies
def get_brain_agent():
    return brain_agent_instance

def get_vector_store():
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
        
        # Create candidate data object
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
