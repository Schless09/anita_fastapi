from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from agents.brain_agent import BrainAgent
from agents.interaction_agent import InteractionAgent
from agents.vector_store import VectorStore
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import httpx
from enum import Enum
import openai
from pydantic import ValidationError
import json
import base64
from openai import OpenAI, AsyncOpenAI
import io
import tempfile
import requests
from typing import Annotated
import aiohttp
import traceback
import asyncio
from PyPDF2 import PdfReader
from pinecone import Pinecone
import phonenumbers
from retell import Retell
import time
import re
from urllib.parse import urlparse
import uuid
import logging
import logging.handlers
from fastapi.responses import JSONResponse, Response
import hmac
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import email

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

# Test logging configuration
logger.info("\n=== Starting Anita AI Recruitment API ===")
logger.info(f"Environment: {os.getenv('VERCEL_ENV', 'development')}")
logger.info(f"Pinecone Jobs Index: {os.getenv('PINECONE_JOBS_INDEX', 'Unknown')}")
logger.info(f"Pinecone Candidates Index: {os.getenv('PINECONE_CANDIDATES_INDEX', 'Unknown')}")
logger.info(f"Pinecone Call Status Index: {os.getenv('PINECONE_CALL_STATUS_INDEX', 'Unknown')}")
logger.info("=====================================\n")

# Load environment variables
load_dotenv()

# Configure Retell
RETELL_CONFIG = {
    'api_key': os.getenv('RETELL_API_KEY'),
    'phone_number': os.getenv('RETELL_FROM_NUMBER', '+16506486990'),
    'webhook_url': os.getenv('RETELL_WEBHOOK_URL', 'https://85ac-2601-645-8000-fe70-4981-3c72-e8ef-490f.ngrok-free.app/webhook/retell')
}

# Configure SendGrid
SENDGRID_CONFIG = {
    'api_key': os.getenv('SENDGRID_API_KEY'),
    'sender_email': os.getenv('SENDGRID_SENDER_EMAIL', 'anita@anita.ai'),
    'webhook_url': os.getenv('SENDGRID_WEBHOOK_URL', 'https://85ac-2601-645-8000-fe70-4981-3c72-e8ef-490f.ngrok-free.app/email/webhook')
}

# Initialize FastAPI app
app = FastAPI(
    title="Anita AI Recruitment API",
    description="API for AI-driven recruitment with enhanced candidate-job matching",
    version="2.0.0"
)

# Mount static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Slack client
SLACK_APP_ID = os.getenv('SLACK_APP_ID')
SLACK_CLIENT_ID = os.getenv('SLACK_CLIENT_ID')
SLACK_CLIENT_SECRET = os.getenv('SLACK_CLIENT_SECRET')
SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL_ID = "C08KAN8AYJJ"  # Updated channel ID

def verify_slack_request(request_body: bytes, timestamp: str, signature: str) -> bool:
    """Verify that the request came from Slack."""
    if not SLACK_SIGNING_SECRET:
        logger.error("Slack signing secret not configured")
        return False

    # Create basestring by concatenating version, timestamp, and body
    basestring = f"v0:{timestamp}:{request_body.decode()}"
    
    # Create signature using HMAC SHA256
    my_signature = 'v0=' + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures using constant time comparison
    return hmac.compare_digest(my_signature, signature)

async def send_slack_notification(message: str) -> bool:
    """
    Send a notification to Slack using the simplified approach.
    Returns True if successful, False otherwise.
    """
    if not SLACK_BOT_TOKEN:
        logger.error("❌ Slack notification failed: Missing SLACK_BOT_TOKEN")
        return False

    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "channel": SLACK_CHANNEL_ID,
        "text": message,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📧 *New Message*\n\n{message}"
                }
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"🔔 Sending Slack notification...")
            logger.info(f"🎯 Target Channel: {SLACK_CHANNEL_ID}")
            logger.info(f"🔑 Using token starting with: {SLACK_BOT_TOKEN[:20]}...")
            
            response = await client.post(url, headers=headers, json=data)
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('ok'):
                logger.info("✅ Slack notification sent successfully")
                return True
            else:
                error = response_data.get('error', 'Unknown error')
                logger.error(f"❌ Slack API error: {error}")
                logger.error(f"📝 Full response: {response_data}")
                if error == 'invalid_auth':
                    logger.error("🔐 Authentication failed. Please check your Bot User OAuth Token")
                elif error == 'channel_not_found':
                    logger.error(f"📢 Channel {SLACK_CHANNEL_ID} not found. Make sure the bot is invited to the channel")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to send Slack notification: {str(e)}")
        logger.error(f"💥 Error type: {type(e)}")
        logger.error(f"📜 Stack trace: {traceback.format_exc()}")
        return False

@app.get("/")
async def root():
    """Root endpoint that returns API information and environment details"""
    environment = "development" if os.getenv("VERCEL_ENV") is None else os.getenv("VERCEL_ENV")
    pinecone_index = os.getenv("PINECONE_JOBS_INDEX", "Unknown")
    
    return {
        "name": "Anita AI Recruitment API",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/docs",
        "environment": environment,
        "vector_db": {
            "name": "Pinecone",
            "jobs_index": pinecone_index,
            "candidates_index": os.getenv("PINECONE_CANDIDATES_INDEX", "Unknown"),
            "call_status_index": os.getenv("PINECONE_CALL_STATUS_INDEX", "Unknown")
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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

# Initialize in-memory storage for job statuses
job_statuses: Dict[str, Dict[str, Any]] = {}

# Initialize in-memory storage for call statuses (will be synced with Pinecone)
call_statuses: Dict[str, Dict[str, Any]] = {}

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Job analysis prompt template
JOB_ANALYSIS_PROMPT = """You are an AI assistant helping a recruiter match candidates to technical roles at startups. For each role provided, extract and structure the data into a standardized JSON format optimized for matching algorithms.

Return Format Requirements:
- Output a valid JSON object format (not a stringified JavaScript object)
- Use double quotes for all keys and string values
- Include all fields listed below. Use "n/a" for missing or unavailable information
- Use two-space indentation for readability
- Do not include trailing commas
- Quote numeric values (e.g., "5+" instead of 5+)
- Return arrays using proper JSON syntax (e.g., ["value1", "value2"])
- Return booleans as true or false (do not quote them)
- Use consistent formats:
  - Salaries and funding: "$200k+", "$50M"
  - Equity: "0.1% - 0.5%"
  - Always return cities and states as arrays, even if only one value (e.g., ["New York"], ["NY"])

Value Standardization Rules:
- Tech Stack Normalization: Standardize technology names (e.g., React.js/ReactJS → React, TensorFlow/TF → TensorFlow)
- Location Standardization: Use consistent "City, ST" formats and return "city" and "state" as arrays
- Requirement Classification:
  - "must_have": Absolute requirements
  - "preferred": Desired but not required
  - "nice_to_have": Beneficial but optional
- Return the following as arrays of strings if multiple items apply:
  - "investors", "tech_stack_must_haves", "tech_stack_nice_to_haves", "tech_stack_tags", "coding_languages_versions", "version_control_experience", "ci_cd_tools", "collaborative_tools", "domain_expertise", "infrastructure_experience", "technical_challenges", "key_responsibilities", "expected_deliverables", "ideal_companies", "deal_breakers", "culture_fit_indicators", "startup_mindset_requirements", "growth_mindset_indicators", "interview_process_tags", "interview_focus_areas"

Required JSON Structure:
{
  "company_name": "string",
  "company_url": "string",
  "company_stage": ["Seed", "Series A", "Series B", "Series C", "Growth"],
  "most_recent_funding_round_amount": "string",
  "total_funding_amount": "string",
  "investors": ["string"],
  "team_size": "string",
  "founding_year": "string",
  "company_mission": "string",
  "target_market": ["B2B", "B2C", "Enterprise", "SMB"],
  "industry_vertical": "string",
  "company_vision": "string",
  "company_growth_story": "string",
  "company_culture": {
    "work_environment": "string",
    "decision_making": "string",
    "collaboration_style": "string",
    "risk_tolerance": "string",
    "values": "string"
  },
  "scaling_plans": "string",
  "mission_and_impact": "string",
  "tech_innovation": "string",
  "job_title": "string",
  "job_url": "string",
  "positions_available": "string",
  "hiring_urgency": ["ASAP", "Within 30 days", "Within 60 days", "Ongoing"],
  "seniority_level": ["1+ years", "3+ years", "5+ years", "7+ years", "10+ years"],
  "work_arrangement": ["Remote", "On-site", "Hybrid"],
  "city": ["string"],
  "state": ["string"],
  "visa_sponsorship": "string",
  "work_authorization": "string",
  "salary_range": "string",
  "equity_range": "string",
  "reporting_structure": "string",
  "team_composition": "string",
  "role_status": "string",
  "role_category": ["SWE", "ML Engineer", "AI Engineer", "Data Engineer", "DevOps"],
  "tech_stack_must_haves": ["string"],
  "tech_stack_nice_to_haves": ["string"],
  "tech_stack_tags": ["string"],
  "tech_breadth_requirement": ["Full-Stack", "Frontend-Leaning", "Backend-Leaning", "ML/AI-Focused"],
  "minimum_years_of_experience": "string",
  "domain_expertise": ["string"],
  "ai_ml_experience": "string",
  "infrastructure_experience": ["string"],
  "system_design_level": "string",
  "coding_proficiency_required": ["Basic", "Intermediate", "Expert"],
  "coding_languages_versions": ["string"],
  "version_control_experience": ["string"],
  "ci_cd_tools": ["string"],
  "collaborative_tools": ["string"],
  "leadership_requirement": ["None", "Preferred", "Required"],
  "education_requirement": "string",
  "advanced_degree_preference": "string",
  "papers_publications_preferred": "string",
  "prior_startup_experience": ["Required", "Preferred", "Not required"],
  "advancement_history_required": boolean,
  "independent_work_capacity": "string",
  "skills_must_have": ["string"],
  "skills_preferred": ["string"],
  "product_details": "string",
  "product_development_stage": ["Prototype", "MVP", "Market-ready", "Scaling"],
  "technical_challenges": ["string"],
  "key_responsibilities": ["string"],
  "scope_of_impact": ["Team", "Department", "Company", "Industry"],
  "expected_deliverables": ["string"],
  "product_development_methodology": ["Agile", "Scrum", "Kanban"],
  "stage_of_codebase": ["Greenfield", "Established", "Legacy Refactoring"],
  "growth_trajectory": "string",
  "founder_background": "string",
  "funding_stability": "string",
  "expected_hours": "string",
  "ideal_companies": ["string"],
  "deal_breakers": ["string"],
  "culture_fit_indicators": ["string"],
  "startup_mindset_requirements": ["string"],
  "autonomy_level_required": "string",
  "growth_mindset_indicators": ["string"],
  "ideal_candidate_profile": "string",
  "interview_process_tags": ["string"],
  "technical_assessment_type": ["string"],
  "interview_focus_areas": ["string"],
  "time_to_hire": "string",
  "decision_makers": ["string"],
  "recruiter_pitch_points": ["string"]
}

Instructions:
1. Extract all relevant data from the information provided.
2. If information is not explicitly stated:
   - Make reasonable inferences based on context, but do not overextend.
   - Use "n/a" for missing or unknown information.
3. For startup tech roles, prioritize identifying:
   - Tech stack and depth
   - Startup experience/mindset
   - Autonomy and ownership
   - Growth and scaling context
4. For ML/AI roles, prioritize identifying:
   - ML frameworks and methods
   - Research vs. applied ML experience
   - Deployment and infrastructure skills
   - Domain-specific AI experience
5. Return complete JSON with all fields populated using "n/a" or reasonable default formatting when necessary."""

# Initialize Pinecone
logger.info("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get list of existing indexes
existing_indexes = [index.name for index in pc.list_indexes()]
logger.info(f"Existing indexes: {existing_indexes}")

# Get index names from environment variables
jobs_index_name = os.getenv('PINECONE_JOBS_INDEX')
candidates_index_name = os.getenv('PINECONE_CANDIDATES_INDEX')
call_status_index_name = os.getenv('PINECONE_CALL_STATUS_INDEX', 'call-statuses')  # Default to call-statuses

# Validate environment variables
if not jobs_index_name:
    raise ValueError("Missing required environment variable: PINECONE_JOBS_INDEX")
if not candidates_index_name:
    raise ValueError("Missing required environment variable: PINECONE_CANDIDATES_INDEX")

# Validate indexes exist
if jobs_index_name not in existing_indexes:
    raise ValueError(f"Jobs index '{jobs_index_name}' does not exist")
if candidates_index_name not in existing_indexes:
    raise ValueError(f"Candidates index '{candidates_index_name}' does not exist")
if call_status_index_name not in existing_indexes:
    raise ValueError(f"Call status index '{call_status_index_name}' does not exist")

# Initialize indexes
job_index = pc.Index(jobs_index_name)
candidates_index = pc.Index(candidates_index_name)
call_status_index = pc.Index(call_status_index_name)

# Log environment info
environment = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Running in {environment} environment")
logger.info(f"Using jobs index: {jobs_index_name}")
logger.info(f"Using candidates index: {candidates_index_name}")
logger.info(f"Using call status index: {call_status_index_name}")

# Initialize agents with the existing Pinecone instances
interaction_agent = InteractionAgent()
vector_store = VectorStore(init_openai=True, existing_indexes={
    'candidates_index': candidates_index,
    'jobs_index': job_index,
    'call_status_index': call_status_index
})
brain_agent = BrainAgent(vector_store)

# Add Retell AI configuration
RETELL_API_KEY = os.getenv('RETELL_API_KEY')
RETELL_API_BASE = "https://api.retellai.com/v2"
RETELL_FROM_NUMBER = os.getenv('RETELL_FROM_NUMBER')
RETELL_AGENT_ID = os.getenv('RETELL_AGENT_ID')
RETELL_PHONE_NUMBER = os.getenv('RETELL_PHONE_NUMBER')

class Location(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None

class TranscriptData(BaseModel):
    candidate_id: str
    transcript: str

class RetellCallStatus(str, Enum):
    CREATED = "created"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    ERROR = "error"
    ERROR_UNKNOWN = "error_unknown"

class RetellCallData(BaseModel):
    candidate_id: str
    call_id: str
    transcript: Optional[str] = None
    email: Optional[str] = None

    @validator('candidate_id')
    def validate_candidate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("candidate_id cannot be empty")
        if not v.startswith('candidate_'):
            raise ValueError("Invalid candidate_id format")
        return v.strip()

    @validator('call_id')
    def validate_call_id(cls, v):
        if not v or not v.strip():
            raise ValueError("call_id cannot be empty")
        if not v.startswith('call_'):
            raise ValueError("Invalid call_id format")
        return v.strip()

class RetellTranscriptResponse(BaseModel):
    call_id: str
    transcript: str
    call_analysis: Optional[Dict[str, Any]]
    call_status: RetellCallStatus
    user_sentiment: Optional[Literal["Positive", "Negative", "Neutral", "Unknown"]] = "Unknown"

class EnhancedTranscript(BaseModel):
    raw_transcript: str
    call_summary: str
    user_sentiment: str
    call_successful: bool
    custom_analysis: Dict[str, Any]
    timestamp: str  # Changed from datetime to str
    call_status: RetellCallStatus
    call_duration: Optional[float]
    error_details: Optional[str]

class CandidateData(BaseModel):
    name: str
    email: str
    phone_number: str
    linkedin: Optional[str] = None
    resume_text: Optional[str] = None  # Changed from resume to resume_text

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @validator('email')
    def validate_email(cls, v):
        if not v or not v.strip():
            raise ValueError('Email cannot be empty')
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.strip()

    @validator('phone_number')
    def validate_phone(cls, v):
        if not v or not v.strip():
            raise ValueError('Phone number cannot be empty')
        # Remove any non-digit characters except '+'
        phone = ''.join(filter(str.isdigit, v.replace('+', '')))
        if not phone.startswith('+'):
            phone = '+' + phone
        return phone

    @validator('linkedin')
    def validate_linkedin(cls, v):
        if v:
            v = v.strip()
            if not v:
                return None
            if not (v.startswith('http://') or v.startswith('https://')):
                v = 'https://' + v
        return v

    @validator('resume_text')
    def validate_resume(cls, v):
        if v:
            v = v.strip()
            if not v:
                return None
            # Truncate if too long (Retell AI might have limits)
            max_length = 5000
            if len(v) > max_length:
                return v[:max_length]
        return v

class JobMatchRequest(BaseModel):
    job_id: str
    top_k: Optional[int] = Field(default=5, gt=0, le=100)

class CompanyCulture(BaseModel):
    work_environment: Optional[str] = "Not specified"
    decision_making: Optional[str] = "Not specified"
    collaboration_style: Optional[str] = "Not specified"
    risk_tolerance: Optional[str] = "Not specified"
    values: Optional[str] = "Not specified"

class CompanyStage(str, Enum):
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C = "Series C"
    GROWTH = "Growth"

class TargetMarket(str, Enum):
    B2B = "B2B"
    B2C = "B2C"
    ENTERPRISE = "Enterprise"
    SMB = "SMB"

class HiringUrgency(str, Enum):
    ASAP = "ASAP"
    WITHIN_30_DAYS = "Within 30 days"
    WITHIN_60_DAYS = "Within 60 days"
    ONGOING = "Ongoing"

class SeniorityLevel(str, Enum):
    ONE_PLUS = "1+ years"
    THREE_PLUS = "3+ years"
    FIVE_PLUS = "5+ years"
    SEVEN_PLUS = "7+ years"
    TEN_PLUS = "10+ years"

class WorkArrangement(str, Enum):
    REMOTE = "Remote"
    ON_SITE = "On-site"
    HYBRID = "Hybrid"

class RoleCategory(str, Enum):
    SWE = "SWE"
    ML_ENGINEER = "ML Engineer"
    AI_ENGINEER = "AI Engineer"
    DATA_ENGINEER = "Data Engineer"
    DEVOPS = "DevOps"

class TechBreadthRequirement(str, Enum):
    FULL_STACK = "Full-Stack"
    FRONTEND_LEANING = "Frontend-Leaning"
    BACKEND_LEANING = "Backend-Leaning"
    ML_AI_FOCUSED = "ML/AI-Focused"

class LeadershipRequirement(str, Enum):
    NONE = "None"
    PREFERRED = "Preferred"
    REQUIRED = "Required"

class ProductDevelopmentStage(str, Enum):
    PROTOTYPE = "Prototype"
    MVP = "MVP"
    MARKET_READY = "Market-ready"
    SCALING = "Scaling"

class ScopeOfImpact(str, Enum):
    TEAM = "Team"
    DEPARTMENT = "Department"
    COMPANY = "Company"
    INDUSTRY = "Industry"

class ProductDevelopmentMethodology(str, Enum):
    AGILE = "Agile"
    SCRUM = "Scrum"
    KANBAN = "Kanban"

class StageOfCodebase(str, Enum):
    GREENFIELD = "Greenfield"
    ESTABLISHED = "Established"
    LEGACY_REFACTORING = "Legacy Refactoring"

class TechnicalAssessmentType(str, Enum):
    TAKE_HOME = "Take-home"
    LIVE_CODING = "Live coding"
    SYSTEM_DESIGN = "System design"
    ML_DESIGN = "ML design"

class JobSubmission(BaseModel):
    raw_text: str

    @validator('raw_text')
    def clean_text(cls, v):
        print(f"\n=== Validating input text ===")
        print(f"Input value type: {type(v)}")
        print(f"Input value length: {len(str(v)) if v else 0}")
        print(f"First 100 chars: {str(v)[:100] if v else 'None'}")
        
        if not v:
            print("Validation Error: Empty input")
            raise ValueError("Text cannot be empty")
        
        # Convert to string if not already
        v = str(v)
        
        # Handle escaped characters first
        v = v.replace("\\'", "'")  # Replace escaped single quotes
        v = v.replace('\\"', '"')  # Replace escaped double quotes
        v = v.replace('\\\\', '\\')  # Handle escaped backslashes
        v = v.replace('\\n', '\n')  # Handle escaped newlines
        v = v.replace('\\t', '\t')  # Handle escaped tabs
        v = v.replace('\\r', '\r')  # Handle escaped carriage returns
        
        # Remove any BOM characters
        v = v.replace('\ufeff', '')
        
        # Remove null bytes and other problematic control characters
        # Keep newlines, tabs, and carriage returns
        v = ''.join(char for char in v if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize newlines
        v = v.replace('\r\n', '\n').replace('\r', '\n')
        
        # Strip whitespace from start and end
        v = v.strip()
        
        print(f"Cleaned value length: {len(v)}")
        print(f"First 100 chars after cleaning: {v[:100]}")
        print("=== Validation complete ===\n")
        
        return v

class MatchResponse(BaseModel):
    status: str
    matches: List[Dict[str, Any]]
    total_matches: int

class RetellCallList(BaseModel):
    calls: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

class ProcessedTranscript(BaseModel):
    candidate_name: str
    linkedin_url: Optional[str]
    contact_information: Dict[str, str]
    date_of_call: str
    current_role: Optional[str]
    years_of_experience: Optional[float]
    skills: List[str]
    preferred_work_environment: Optional[str]
    preferred_locations: List[Dict[str, str]]
    minimum_salary: Optional[float]
    work_authorization: Optional[str]
    education: Optional[Dict[str, str]]
    availability: Optional[str]
    interests: Optional[List[str]]
    raw_transcript: str

class MakeCallRequest(BaseModel):
    name: str
    email: str
    phone_number: str
    linkedin: Optional[str] = None
    resume_text: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v) > 100:
            raise ValueError('Name must be between 1 and 100 characters')
        return v

    @validator('phone_number')
    def validate_phone(cls, v):
        if not v:
            raise ValueError('Phone number is required')
        # Remove any non-digit characters except '+'
        phone = '+' + ''.join(filter(str.isdigit, v.replace('+', '')))
        if len(phone) < 10 or len(phone) > 15:
            raise ValueError('Phone number must be between 10 and 15 digits')
        return phone

    @validator('linkedin')
    def validate_linkedin(cls, v):
        if v and len(v) > 255:
            raise ValueError('LinkedIn URL must be less than 255 characters')
        return v or ""

class CallStatusRequest(BaseModel):
    """Request model for checking call status"""
    candidate_id: str
    call_id: str

class RetellWebhookCall(BaseModel):
    call_id: str
    call_status: str
    metadata: Dict[str, Any] = {}
    transcript: Optional[str] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    disconnection_reason: Optional[str] = None
    transcript_object: Optional[Dict[str, Any]] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    direction: Optional[str] = None
    agent_id: Optional[str] = None

class RetellWebhookPayload(BaseModel):
    event: str
    call: RetellWebhookCall

async def process_pdf_to_text(file: UploadFile) -> Dict[str, Any]:
    """Process a PDF file and extract its text content."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Validate PDF content
        if not contents.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid PDF file format. The file does not appear to be a valid PDF."
            )
            
        # Create a temporary file to handle the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(contents)
            temp_file.flush()
            
            try:
                # Read PDF with PyPDF2
                pdf_reader = PdfReader(temp_file.name)
                
                # Validate PDF structure
                if len(pdf_reader.pages) == 0:
                    raise HTTPException(status_code=400, detail="PDF file is empty")
                
                # Extract text from all pages
                text_content = []
                image_pages = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        # Try to extract text
                        text = page.extract_text()
                        
                        # Check if page contains mostly images
                        if text and text.strip():
                            # If we got text, add it
                            text_content.append(text)
                        else:
                            # If no text, mark as image page
                            image_pages.append(page_num)
                            
                    except Exception as page_error:
                        print(f"Error extracting text from page {page_num}: {str(page_error)}")
                        image_pages.append(page_num)
                        continue
                
                # Combine all text
                combined_text = "\n\n".join(text_content)
                
                # If we have image pages, add a warning
                if image_pages:
                    print(f"Warning: Pages {image_pages} appear to be image-based and may not be fully processed")
                
                # If we got no text at all, raise an error
                if not text_content:
                    raise HTTPException(
                        status_code=400, 
                        detail="Your resume is unreadable and may contain too many images. Please submit a text-based resume."
                    )
                
                # Calculate text quality metrics
                total_pages = len(pdf_reader.pages)
                text_pages = total_pages - len(image_pages)
                text_quality = (text_pages / total_pages) * 100 if total_pages > 0 else 0
                
                return {
                    "text": combined_text,
                    "filename": file.filename,
                    "page_count": total_pages,
                    "text_pages": text_pages,
                    "image_pages": image_pages,
                    "text_quality": text_quality,
                    "warning": f"Pages {image_pages} appear to be image-based" if image_pages else None
                }
                    
            except Exception as pdf_error:
                print(f"Error reading PDF: {str(pdf_error)}")
                raise HTTPException(
                    status_code=400,
                    detail="Your resume is unreadable and may contain too many images. Please submit a text-based resume."
                )
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up temporary file: {str(cleanup_error)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Your resume is unreadable and may contain too many images. Please submit a text-based resume."
        )
    finally:
        await file.seek(0)

async def process_resume_text(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process resume using OpenAI GPT-4 Turbo to extract key information."""
    if not OPENAI_API_KEY:
        print("Warning: OpenAI API key not configured, skipping resume processing")
        return {
            "raw_text": resume_data.get("text", ""),
            "processed": False,
            "error": "OpenAI API key not configured"
        }
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare the content for GPT-4
        content = resume_data.get("text", "").strip()
        candidate_id = resume_data.get("candidate_id")
        
        if not content:
            return {
                "raw_text": resume_data.get("text", ""),
                "processed": False,
                "error": "No content available for processing"
            }
        
        system_message = """You are an expert resume analyzer. Extract the following information in a structured format:
        - skills: List of technical and soft skills
        - experience: List of work experiences with company, role, and duration
        - education: List of educational qualifications
        - achievements: List of key achievements
        - summary: A brief professional summary
        - current_company: The candidate's current company (most recent position). If not currently employed, return null
        - current_role: The candidate's current role/title. If not currently employed, return null

        Format the response as a valid JSON object with these exact keys. For current_company and current_role, 
        look for indicators like "present", "current", or most recent dates to determine the current position."""

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Please analyze this resume and extract the key information:\n\n{content}"}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={ "type": "json_object" },
                stream=True  # Enable streaming
            )
            
            # Collect the streamed response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            
            try:
                processed_data = json.loads(full_response)
                
                # If we have a candidate_id, update the vector store with the processed data
                if candidate_id:
                    try:
                        vector_store = VectorStore(init_openai=True)
                        # Update the candidate's profile with the processed data
                        vector_store.update_candidate_profile(candidate_id, {
                            "processed_resume": processed_data,
                            "processed_at": datetime.utcnow().isoformat()
                        })
                    except Exception as store_error:
                        print(f"Warning: Failed to store processed resume data: {str(store_error)}")
                
                return {
                    "raw_text": resume_data.get("text", ""),
                    "processed": True,
                    "structured_data": processed_data,
                    "processed_at": datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError as e:
                print(f"Error parsing OpenAI response: {str(e)}")
                return {
                    "raw_text": resume_data.get("text", ""),
                    "processed": True,
                    "structured_data": {
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "achievements": [],
                        "summary": "Failed to parse structured data from resume",
                        "current_company": None,
                        "current_role": None
                    },
                    "error": f"Failed to parse OpenAI response: {str(e)}",
                    "processed_at": datetime.utcnow().isoformat()
                }
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return {
                "raw_text": resume_data.get("text", ""),
                "processed": False,
                "error": f"OpenAI API error: {str(api_error)}"
            }
            
    except Exception as e:
        print(f"Error processing resume with OpenAI: {str(e)}")
        return {
            "raw_text": resume_data.get("text", ""),
            "processed": False,
            "error": str(e)
        }

@app.post("/candidates")
async def submit_candidate(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    resume: UploadFile = File(...),
    linkedin: Optional[str] = Form(None)
):
    try:
        # Generate a unique candidate ID
        candidate_id = f"candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n=== Processing candidate submission {candidate_id} ===")
        
        # Process the PDF file - this is quick and needed for the call
        try:
            pdf_result = await process_pdf_to_text(resume)
            resume_text = pdf_result["text"]
            print("Successfully extracted text from resume")
        except HTTPException as pdf_error:
            raise HTTPException(
                status_code=pdf_error.status_code,
                detail=f"Failed to process resume: {pdf_error.detail}"
            )

        # Create basic candidate profile for initial storage
        profile = {
            "id": candidate_id,
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "linkedin": linkedin,
            "resume_text": resume_text,
            "status": "processing",
            "submitted_at": datetime.utcnow().isoformat()
        }
        print("Created candidate profile")

        # Store basic profile in vector database
        try:
            print("Initializing VectorStore with OpenAI support...")
            vector_store = VectorStore(init_openai=True)
            print("Storing candidate in Pinecone...")
            vector_result = vector_store.store_candidate(candidate_id, profile)
            if vector_result.get('status') == 'error':
                print(f"Error storing candidate in vector database: {vector_result.get('message')}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to store candidate in vector database: {vector_result.get('message')}"
                )
            print("Successfully stored candidate in Pinecone")
        except Exception as e:
            print(f"Error storing candidate: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store candidate: {str(e)}"
            )

        # Create a copy of the resume file for the make_call function
        resume_copy = UploadFile(
            filename=resume.filename,
            file=io.BytesIO(await resume.read())
        )
        await resume.seek(0)

        # Trigger the makeCall endpoint
        try:
            print(f"Initiating call with candidate {candidate_id}...")
            make_call_response = await make_call(
                candidate_id=candidate_id,
                name=name,
                email=email,
                phone_number=phone_number,
                linkedin=linkedin or "",
                resume=resume_copy
            )
            
            # Add resume processing to background tasks
            background_tasks.add_task(process_candidate_resume, candidate_id, resume_text)
            
            return {
                "status": "success",
                "message": "Candidate profile created and call initiated successfully",
                "candidate_id": candidate_id,
                "profile": profile,
                "vector_store_status": vector_result.get('status', 'unknown'),
                "call_data": make_call_response
            }
            
        except Exception as call_error:
            print(f"Warning: Failed to initiate call: {str(call_error)}")
            return {
                "status": "partial_success",
                "message": "Candidate profile created but failed to initiate call",
                "candidate_id": candidate_id,
                "profile": profile,
                "vector_store_status": vector_result.get('status', 'unknown'),
                "error": str(call_error)
            }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in submit_candidate: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        print("=== Candidate submission processing complete ===\n")

async def process_candidate_resume(candidate_id: str, resume_text: str):
    """Process candidate resume in the background."""
    try:
        print(f"\n=== Processing resume for candidate {candidate_id} ===")
        
        # Process resume with OpenAI
        resume_data = {
            "text": resume_text,
            "candidate_id": candidate_id
        }
        processed_resume = await process_resume_text(resume_data)
        
        # Update candidate profile with processed data
        vector_store.update_candidate_profile(
            candidate_id,
            {
                "processed_resume": processed_resume,
                "status": "completed",
                "processed_at": datetime.utcnow().isoformat()
            }
        )
        
        print(f"Successfully processed resume for candidate {candidate_id}")
        
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        vector_store.update_candidate_profile(
            candidate_id,
            {
                "status": "failed",
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }
        )

async def process_transcript_with_openai(transcript: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process transcript with OpenAI to generate summary and extract structured data."""
    try:
        # Extract first name from metadata
        candidate_name = metadata.get('name', '')
        first_name = candidate_name.split()[0] if candidate_name else ''

        # Create the prompt with the transcript
        prompt = f"""Please analyze this job interview transcript and provide a structured summary.
        
        Transcript:
        {transcript}
        
        Please extract and provide the following information in a structured format:
        
        1. Key Information (required for matching):
        - Skills: List all technical and soft skills mentioned
        - Years of Experience: Total years of relevant experience
        - Preferred Work Environment: Remote/Hybrid/On-site preferences
        - Preferred Locations: List of cities/regions they're interested in
        - Minimum Salary: Expected or minimum salary requirement
        - Work Authorization: Current work authorization status
        
        2. Additional Information:
        - Current Role: Details about current position
        - Career Goals: Career objectives and aspirations
        - Notable Projects: Key projects or achievements
        - Leadership Experience: Any leadership or management experience
        - Education: Educational background
        - Industry Preferences: Preferred industries or company types
        
        3. Summary for Email:
        Hi {first_name},
        
        Thank you for taking the time to speak with me today! Here are the key points from our conversation:
        
        [List 3-5 key points from the conversation]
        
        [Include any specific next steps or action items discussed]
        
        Please provide the response in JSON format with these exact keys:
        {{
            "skills": [],
            "years_of_experience": "",
            "preferred_work_environment": "",
            "preferred_locations": [],
            "minimum_salary": "",
            "work_authorization": "",
            "current_role": "",
            "career_goals": "",
            "notable_projects": "",
            "leadership_experience": "",
            "education": "",
            "industry_preferences": "",
            "key_points": [],
            "next_steps": "",
            "email_summary": ""
        }}
        """

        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a professional recruiter analyzing a job interview. Extract all required information accurately and format it as specified."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format={ "type": "json_object" }
        )

        # Parse the JSON response
        try:
            processed_data = json.loads(response.choices[0].message.content)
            # Add first name to processed data
            processed_data['first_name'] = first_name
            processed_data['timestamp'] = datetime.now().isoformat()
            return processed_data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI response as JSON: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error processing transcript with OpenAI: {str(e)}")
        raise

async def store_in_pinecone(candidate_id: str, processed_data: dict):
    """Store processed transcript data in Pinecone"""
    try:
        vector_store.update_candidate_profile(
            candidate_id,
            {
                'processed_transcript': processed_data,
                'processed_at': datetime.now().isoformat()
            }
        )
        logger.info(f"✅ Successfully stored data in Pinecone for candidate {candidate_id}")
    except Exception as e:
        logger.error(f"❌ Error storing in Pinecone: {str(e)}")
        raise

@app.post("/candidate/transcript", response_model=Dict[str, Any])
async def add_transcript(transcript_data: TranscriptData):
    """
    Add and process a transcript for a candidate profile.
    
    The transcript will be:
    1. Processed using OpenAI to extract structured information
    2. Used to enhance the candidate's profile
    3. Potentially trigger re-matching if the candidate was previously unmatched
    """
    try:
        # Process transcript with OpenAI
        processed_data = await process_transcript_with_openai(transcript_data.transcript, {
            "name": transcript_data.candidate_id.split('_')[1]
        })
        
        # Update the candidate profile with both raw and processed transcript
        result = brain_agent.add_transcript_to_profile(
            transcript_data.candidate_id,
            {
                "processed_data": processed_data,
                "raw_transcript": transcript_data.transcript
            }
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return {
            "status": "success",
            "message": "Transcript processed and stored successfully",
            "processed_data": processed_data,
            "candidate_state": result.get('current_state')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process transcript: {str(e)}"
        )

@app.post("/jobs/match-candidates", response_model=MatchResponse)
async def match_candidates_to_job(request: JobMatchRequest):
    """
    Find candidates that match a specific job posting.
    
    Enhanced matching considers:
    - Semantic similarity of skills and experience
    - Location preferences
    - Work environment preferences
    - Compensation requirements
    - Work authorization requirements
    
    Returns detailed match information including match scores and reasons.
    """
    result = brain_agent.find_similar_candidates(request.job_id, request.top_k)
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['message'])
    
    return {
        "status": "success",
        "matches": result['matches'],
        "total_matches": len(result['matches'])
    }

@app.post("/test-email/{call_id}")
async def test_email(call_id: str):
    """Test endpoint to send a summary email for a specific call."""
    try:
        # Process the transcript and send email
        processed_data = await fetch_and_store_retell_transcript(call_id)
        
        return {
            "status": "success",
            "message": "Email sent successfully",
            "processed_data": processed_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def validate_retell_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and process the Retell AI response"""
    if not response_data.get('transcript'):
        raise ValueError("No transcript found in Retell AI response")

    call_status = response_data.get('call_status')
    if not call_status:
        raise ValueError("No call status found in Retell AI response")

    if call_status not in RetellCallStatus.__members__.values():
        raise ValueError(f"Invalid call status: {call_status}")

    if call_status != RetellCallStatus.ENDED:
        raise ValueError(f"Call is not completed. Current status: {call_status}")

    return response_data

async def sync_call_statuses():
    """Sync call statuses from Pinecone to memory on startup"""
    try:
        try:
            query_response = call_status_index.query(
                vector=[0] * 1536,  # Dummy vector since we're just getting metadata
                top_k=100,  # Adjust based on your needs
                include_metadata=True
            )
            
            if query_response.matches:
                for match in query_response.matches:
                    try:
                        call_id = match.id
                        metadata = match.metadata
                        call_statuses[call_id] = metadata
                    except Exception as match_error:
                        logger.error(f"Error processing match for call {match.id}: {str(match_error)}")
                        continue
                
            logger.info(f"Synced {len(call_statuses)} call statuses from Pinecone")
                
        except Exception as query_error:
            logger.error(f"Error querying Pinecone: {str(query_error)}")
            raise
            
    except Exception as e:
        logger.error(f"Critical error in sync_call_statuses: {str(e)}")
        # Don't raise here as this is a startup function

def sanitize_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata to be compatible with Pinecone requirements"""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            sanitized[key] = value
        elif isinstance(value, dict):
            # Flatten nested dictionary with dot notation
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)):
                    sanitized[f"{key}.{nested_key}"] = nested_value
    return sanitized

async def update_call_status(call_id: str, status_data: Dict[str, Any]):
    """Update call status in both memory and Pinecone"""
    try:
        logger.info(f"\n=== Updating call status for {call_id} ===")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
        logger.info(f"Status data: {json.dumps(status_data, indent=2)}")
        
        # Validate status data
        if not isinstance(status_data, dict):
            raise ValueError(f"Invalid status_data type: {type(status_data)}")
        
        # Update in-memory storage
        try:
            call_statuses[call_id] = status_data
            logger.info(f"Updated in-memory status for call {call_id}")
        except Exception as memory_error:
            logger.error(f"Error updating in-memory status: {str(memory_error)}")
            raise
        
        # Update in Pinecone
        try:
            # Create a vector with one non-zero value
            dummy_vector = [0.0] * 1536
            dummy_vector[0] = 1.0  # Set first value to 1.0
            
            # Sanitize metadata for Pinecone
            metadata = sanitize_metadata(status_data)
            metadata['call_id'] = call_id  # Ensure call_id is in metadata for querying
            
            logger.info(f"Sanitized metadata for Pinecone: {json.dumps(metadata, indent=2)}")
            
            call_status_index.upsert(vectors=[(
                call_id,
                dummy_vector,  # Vector with one non-zero value
                metadata
            )])
            logger.info(f"Successfully updated Pinecone for call {call_id}")
        except Exception as pinecone_error:
            logger.error(f"Error updating Pinecone: {str(pinecone_error)}")
            logger.error(f"Error type: {type(pinecone_error)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            # Try to rollback memory update
            if call_id in call_statuses:
                del call_statuses[call_id]
            raise
        
        logger.info(f"Successfully updated call status for {call_id} in both memory and Pinecone")
        
    except Exception as e:
        logger.error(f"Critical error in update_call_status: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        raise

@app.post("/calls/{call_id}/update-email")
async def update_call_email(call_id: str, email: str):
    """Update the email address for a call and trigger email sending."""
    try:
        print(f"\n=== Updating email for call {call_id} ===")
        
        if call_id not in call_statuses:
            raise HTTPException(
                status_code=404,
                detail=f"Call {call_id} not found"
            )
            
        # Update call status with new email
        status_data = call_statuses[call_id]
        status_data['candidate_email'] = email
        status_data['email_sent'] = False  # Reset email sent flag
        status_data['last_updated'] = datetime.utcnow().isoformat()
        
        await update_call_status(call_id, status_data)
        
        # Process the transcript and send email
        await fetch_and_store_retell_transcript(call_id)
        
        return {
            "status": "success",
            "message": f"Email updated and transcript processing triggered for call {call_id}",
            "call_status": status_data
        }
        
    except Exception as e:
        print(f"Error updating email: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update email: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await sync_call_statuses()
        num_removed = clean_test_calls()
        if num_removed > 0:
            logger.info(f"Removed {num_removed} test calls")
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    try:
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

@app.post("/api/makeCall")
async def make_call(
    candidate_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    linkedin: str = Form(None),
    resume: UploadFile = File(...)
):
    """Create a new Retell AI call for a candidate"""
    try:
        print(f"\n=== Creating New Call for Candidate {name} ({candidate_id}) ===")
        print(f"Contact Info: {email} | {phone_number}")
        if linkedin:
            print(f"LinkedIn: {linkedin}")

        # Validate API configuration
        if not RETELL_API_KEY or not RETELL_AGENT_ID or not RETELL_FROM_NUMBER:
            print("ERROR: Missing Retell AI configuration")
            raise HTTPException(
                status_code=500,
                detail="Retell AI configuration is missing. Please check your environment variables."
            )

        # Format phone number to E.164 format
        try:
            parsed_number = phonenumbers.parse(phone_number)
            if not phonenumbers.is_valid_number(parsed_number):
                raise ValueError("Invalid phone number")
            formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
            print(f"Formatted phone number: {formatted_number}")
        except Exception as e:
            print(f"ERROR: Invalid phone number format: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phone number format: {str(e)}"
            )

        # Validate resume file
        if not resume.filename.lower().endswith('.pdf'):
            print(f"ERROR: Invalid resume format: {resume.filename}")
            raise HTTPException(
                status_code=400,
                detail="Resume must be a PDF file"
            )
        print(f"Resume file: {resume.filename}")

        # Quick PDF text extraction for basic dynamic variables
        pdf_result = await process_pdf_to_text(resume)
        resume_text = pdf_result["text"]
        
        # Extract basic information from resume text for dynamic variables
        # This is a quick extraction without OpenAI processing
        current_company = ""
        current_role = ""
        
        # Look for common patterns in the resume text
        lines = resume_text.split('\n')
        for line in lines:
            # Look for current position indicators
            if any(indicator in line.lower() for indicator in ['present', 'current', 'now']):
                parts = line.split('|')
                if len(parts) >= 2:
                    current_role = parts[0].strip()
                    current_company = parts[1].strip()
                    break
        
        # Create a basic summary for dynamic variables
        resume_summary = f"""
        Name: {name}
        Current Role: {current_role or 'Not specified'}
        Current Company: {current_company or 'Not specified'}
        Email: {email}
        Phone: {formatted_number}
        LinkedIn: {linkedin or 'Not provided'}
        """.strip()

        # Prepare the request object
        retell_payload = {
            "from_number": RETELL_FROM_NUMBER,
            "to_number": formatted_number,
            "agent_id": RETELL_AGENT_ID,
            "metadata": {
                "candidate_id": candidate_id,
                "name": name,
                "email": email,
                "linkedin": linkedin,
                "current_company": current_company
            },
            "retell_llm_dynamic_variables": {
                "first_name": name.split()[0],  # Get first name
                "last_name": name.split()[-1] if len(name.split()) > 1 else "",  # Get last name if exists
                "current_company": current_company or "",  # Use extracted current company or empty string
                "email": email,
                "phone_number": formatted_number,
                "resume_summary": resume_summary  # Add the basic resume information
            }
        }

        print(f"\nMaking call with payload: {json.dumps(retell_payload, indent=2)}")

        # Make the call
        async with httpx.AsyncClient(verify=True) as client:
            print(f"Sending request to {RETELL_API_BASE}/create-phone-call")
            response = await client.post(
                f"{RETELL_API_BASE}/create-phone-call",
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=retell_payload
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            if response.status_code not in (200, 201):
                error_detail = response.json() if response.text else "No error details available"
                print(f"ERROR creating call: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to create call: {error_detail}"
                )

            call_data = response.json()
            call_id = call_data.get("call_id")
            
            if not call_id:
                print("ERROR: No call_id in response")
                raise HTTPException(
                    status_code=500,
                    detail="No call_id in response"
                )

            print(f"\nSuccessfully created call with ID: {call_id}")

            # Register call status in both memory and Pinecone
            status_data = {
                "status": "registered",
                "candidate_id": candidate_id,
                "timestamp": datetime.utcnow().isoformat(),
                "candidate_name": name,
                "candidate_email": email,
                "current_company": current_company,
                "resume_text": resume_text  # Store the raw resume text for later processing
            }
            await update_call_status(call_id, status_data)
            print(f"Registered call status: {json.dumps(status_data, indent=2)}")

            return {
                "message": "Call initiated successfully",
                "call_id": call_id,
                "status": "registered",
                "current_company": current_company
            }

    except Exception as e:
        print(f"ERROR in make_call: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create call: {str(e)}"
        )
    finally:
        print("=== Call Creation Complete ===\n")


async def check_call_status(call_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Check the status of a call with Retell AI."""
    try:
        async with httpx.AsyncClient(verify=True) as client:
            url = f"https://api.retellai.com/v2/get-call/{call_id}"
            print(f"Making request to: {url}")
            
            try:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}"
                    }
                )
                
                print(f"Response status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    call_info = response.json()
                    status = call_info.get('call_status', 'unknown')
                    print(f"Retrieved call status: {status}")
                    print(f"Call info: {json.dumps(call_info, indent=2)}")
                    return True, status, call_info
                elif response.status_code == 404:
                    print(f"Call {call_id} not found")
                    return False, "not_found", {}
                elif response.status_code == 401:
                    print("Authentication failed - invalid API key")
                    return False, "auth_error", {}
                else:
                    print(f"Error response: {response.text}")
                    return False, f"error_{response.status_code}", {}
            except httpx.TimeoutException:
                print(f"Timeout while checking call {call_id}")
                return False, "timeout", {}
            except httpx.RequestError as req_error:
                print(f"Request error: {str(req_error)}")
                return False, "request_error", {}
    except Exception as e:
        print(f"Unexpected error checking call status: {str(e)}")
        return False, "error", {"error": str(e)}

def log_webhook(call_id: str, data: dict):
    """Log webhook data to Pinecone instead of filesystem"""
    try:
        # Create a vector with one non-zero value
        dummy_vector = [0.0] * 1536
        dummy_vector[0] = 1.0
        
        # Store webhook data in Pinecone
        call_status_index.upsert(vectors=[(
            f"webhook_{call_id}",
            dummy_vector,
            {
                "type": "webhook",
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
        )])
        logger.info(f"✅ Successfully logged webhook data for call {call_id}")
    except Exception as e:
        logger.error(f"❌ Error logging webhook data: {str(e)}")

@app.post("/webhook/retell")
async def retell_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming webhooks from Retell."""
    try:
        # Get the raw request body
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Log headers and body for debugging
        logger.info("\n=== Webhook Request Details ===")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Raw body: {body_str}")
        
        # Get the signature from headers
        signature = request.headers.get("x-retell-signature")
        if not signature:
            logger.error("❌ Missing x-retell-signature header")
            return JSONResponse(
                status_code=401,
                content={"error": "Missing Retell signature header"}
            )
        
        # Get API key from environment
        api_key = os.getenv("RETELL_API_KEY")
        if not api_key:
            logger.error("❌ Missing RETELL_API_KEY environment variable")
            return JSONResponse(
                status_code=500,
                content={"error": "Server configuration error"}
            )
        
        # Parse the JSON body
        try:
            data = json.loads(body_str)
            logger.info(f"📝 Parsed webhook data: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON payload: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON payload"}
            )
        
        # Use Retell SDK for signature verification
        retell = Retell(api_key=api_key)
        try:
            valid_signature = retell.verify(
                json.dumps(data, separators=(",", ":"), ensure_ascii=False),
                api_key=api_key,
                signature=signature,
            )
            
            if not valid_signature:
                logger.error("❌ Invalid signature")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid signature"}
                )
        except Exception as e:
            logger.error(f"❌ Error verifying signature: {str(e)}")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid signature"}
            )
        
        # Handle different event types
        event = data.get("event")
        call = data.get("call")
        
        if not event or not call:
            logger.error("❌ Missing required fields in webhook payload")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required fields"}
            )
        
        logger.info(f"📞 Processing {event} event for call {call.get('call_id')}")
        
        # Process the event based on type
        if event == "call_started":
            logger.info(f"📞 Call started: {call.get('call_id')}")
        elif event == "call_ended":
            logger.info(f"📞 Call ended: {call.get('call_id')}")
        elif event == "call_analyzed":
            call_id = call.get('call_id')
            logger.info(f"📝 Call analyzed: {call_id}")
            
            # Process transcript immediately in background task
            logger.info(f"🔄 Adding transcript processing to background tasks for call {call_id}")
            background_tasks.add_task(fetch_and_store_retell_transcript, call_id)
            
            # Return immediately to acknowledge webhook
            logger.info(f"✅ Webhook processed successfully for call {call_id}")
            return Response(status_code=202)  # Accepted
        else:
            logger.warning(f"⚠️ Unknown event type: {event}")
        
        # Return success response
        logger.info("✅ Webhook processed successfully")
        return Response(
            status_code=204,
            content=None,
            media_type=None
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/api/cron/process-transcripts")
async def process_pending_transcripts(request: Request):
    """
    Endpoint for Vercel cron job to process transcripts.
    This endpoint should be called every 60 seconds by a Vercel cron job.
    """
    try:
        # In production, verify the request is from Vercel
        if os.getenv("VERCEL_ENV"):
            vercel_cron_header = request.headers.get("x-vercel-cron")
            if not vercel_cron_header:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized - Missing Vercel cron header"
                )
        
        # Find calls that need processing
        pending_calls = {
            call_id: data for call_id, data in call_statuses.items()
            if data.get('status') == 'needs_processing' and not data.get('processed')
        }
        
        if not pending_calls:
            return {"status": "success", "message": "No pending transcripts to process"}
        
        processed_count = 0
        errors = []
        
        # Process each pending call
        for call_id, data in pending_calls.items():
            try:
                # Process one transcript at a time to stay within time limits
                processed_data = await fetch_and_store_retell_transcript(call_id)
                
                # Update status
                data['processed'] = True
                data['processed_at'] = datetime.utcnow().isoformat()
                data['status'] = 'completed'
                await update_call_status(call_id, data)
                
                processed_count += 1
                
                # Break after processing one to stay within time limits
                # Next cron run will handle remaining transcripts
                break
                
            except Exception as e:
                errors.append(f"Error processing call {call_id}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "remaining_count": len(pending_calls) - processed_count,
            "errors": errors
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing transcripts: {str(e)}"
        )

def clean_test_calls() -> int:
    """Remove test calls from call_statuses dictionary."""
    removed_count = 0
    test_calls = []
    
    # Identify test calls
    for call_id, call_data in call_statuses.items():
        if is_test_call(call_data):
            test_calls.append(call_id)
            
    # Remove test calls
    for call_id in test_calls:
        try:
            del call_statuses[call_id]
            removed_count += 1
        except KeyError:
            continue
            
    return removed_count

def is_test_call(call_data: dict) -> bool:
    """Check if a call is a test call based on certain criteria."""
    if not call_data:
        return False
        
    # Check for test indicators in various fields
    test_indicators = [
        call_data.get('candidate_email', '').lower().startswith('test@'),
        'test' in call_data.get('candidate_name', '').lower(),
        'test' in call_data.get('candidate_id', '').lower(),
        call_data.get('source_id', '').startswith('test_'),
        call_data.get('call_id', '').startswith('test_'),
        'test' in call_data.get('email', '').lower(),
        call_data.get('call_id') == 'call_4d59ecbb9689005c0a3560c9d9e'  # Explicitly mark this call as test
    ]
    
    # Also check if any field contains the word "test" case-insensitively
    for value in call_data.values():
        if isinstance(value, str) and 'test' in value.lower():
            test_indicators.append(True)
            break
    
    return any(test_indicators)

async def cleanup_completed_calls():
    """Background task to process completed calls."""
    try:
        logger.info("🔄 Starting cleanup of completed calls")
        
        # Get unprocessed calls
        unprocessed_calls = {
            call_id: status_data
            for call_id, status_data in call_statuses.items()
            if status_data.get('status') == 'ended' and not status_data.get('processed_by_system')
        }
        
        if not unprocessed_calls:
            logger.info("✅ No unprocessed calls found")
            return
            
        logger.info(f"📝 Found {len(unprocessed_calls)} unprocessed calls")
        
        # Process each unprocessed call
        for call_id, status_data in unprocessed_calls.items():
            try:
                logger.info(f"🔄 Processing call {call_id}")
                
                # Process the transcript
                processed_data = await fetch_and_store_retell_transcript(call_id)
                
                # Update status
                status_data['processed_by_system'] = True
                status_data['processed_at'] = datetime.now().isoformat()
                status_data['processed_data'] = processed_data
                
                # Update call status
                await update_call_status(call_id, status_data)
                logger.info(f"✅ Successfully processed call {call_id}")
                
            except Exception as e:
                logger.error(f"❌ Error processing call {call_id}: {str(e)}")
                continue
                
        logger.info("✅ Completed cleanup of calls")
        
    except Exception as e:
        logger.error(f"❌ Error in cleanup task: {str(e)}")

@app.post("/api/cron/cleanup-calls")
async def cron_cleanup_calls(request: Request):
    """
    Endpoint for Vercel cron job to process completed calls.
    This endpoint should be called every 60 seconds by a Vercel cron job.
    """
    # In production, verify the request is from Vercel
    if os.getenv("VERCEL_ENV"):
        vercel_cron_header = request.headers.get("x-vercel-cron")
        if not vercel_cron_header:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized - Missing Vercel cron header"
            )
    
    return await cleanup_completed_calls()

@app.get("/jobs/most-recent")
async def get_most_recent_job():
    """Get the most recent job posting from the Pinecone index."""
    try:
        # Query the index for the most recent job
        query_response = job_index.query(
            vector=[0] * 1536,  # Dummy vector since we're just getting metadata
            top_k=1,
            include_metadata=True
        )
        
        if not query_response.matches:
            raise HTTPException(
                status_code=404,
                detail="No jobs found in the database"
            )
        
        # Get the most recent job's metadata
        most_recent_job = query_response.matches[0].metadata
        
        return {
            "status": "success",
            "job": most_recent_job
        }
        
    except Exception as e:
        print(f"Error getting most recent job: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get most recent job: {str(e)}"
        )

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a job posting by ID."""
    try:
        print(f"\n=== Attempting to retrieve job {job_id} ===")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        
        # Query Pinecone for the job
        print("Querying Pinecone for job data...")
        query_response = job_index.query(
            vector=[0] * 1536,  # Dummy vector since we're querying by metadata
            filter={"job_id": job_id},
            top_k=1,
            include_metadata=True
        )
        print(f"Pinecone query response: {query_response}")
        
        if not query_response.matches:
            print(f"No matches found for job_id: {job_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Failed to get job: No job found with ID: {job_id}"
            )
        
        # Extract job data from the first match
        print("Extracting job data from Pinecone response...")
        job_data = query_response.matches[0].metadata
        print(f"Found job data: {job_data}")
        
        print("=== Job retrieval successful ===\n")
        return job_data
        
    except Exception as e:
        print(f"\nERROR in get_job: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job: {str(e)}"
        )

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    if job_id not in job_statuses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    return job_statuses[job_id]

async def process_job_in_background(job_id: str, file_content: bytes, filename: str):
    try:
        # Update job status to processing
        job_statuses[job_id] = {
            "status": JobStatus.PROCESSING,
            "progress": 0,
            "message": "Starting job processing"
        }

        # Process the job file
        result = {
            "job_id": job_id,
            "filename": filename,
            "processed_at": datetime.utcnow().isoformat(),
            "file_size": len(file_content),
            "status": "completed"
        }

        # Update job status to completed
        job_statuses[job_id] = {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "message": "Job processing completed successfully",
            "result": result
        }
    except Exception as e:
        # Update job status to failed
        job_statuses[job_id] = {
            "status": JobStatus.FAILED,
            "progress": 0,
            "message": f"Job processing failed: {str(e)}",
            "error": str(e)
        }
        raise

async def fetch_retell_transcript(call_id: str) -> dict:
    """Fetch transcript from Retell API."""
    try:
        # First get call data to check status
        call_data = await fetch_retell_call_data(call_id)
        logger.info(f"📞 Call metadata: {call_data.get('metadata', {})}")
        
        # Check if call was answered but hung up immediately
        if call_data.get('status') == 'ended' and not call_data.get('transcript'):
            logger.info(f"📞 Call {call_id} was answered but hung up immediately")
            return {
                'call_id': call_id,
                'status': 'ended',
                'metadata': call_data.get('metadata', {}),
                'transcript': None,
                'duration': call_data.get('duration', 0)
            }
        
        # If call was unanswered, return early
        if call_data.get('status') == 'unanswered':
            logger.info(f"📞 Call {call_id} was unanswered")
            return {
                'call_id': call_id,
                'status': 'unanswered',
                'metadata': call_data.get('metadata', {}),
                'transcript': None,
                'duration': 0
            }
        
        # Get transcript
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{RETELL_API_BASE}/phone-call/{call_id}/transcript",
                headers={"Authorization": f"Bearer {RETELL_API_KEY}"}
            )
            
            if response.status_code == 404:
                logger.warning(f"⚠️ No transcript available for call {call_id}")
                return {
                    'call_id': call_id,
                    'status': call_data.get('status', 'unknown'),
                    'metadata': call_data.get('metadata', {}),
                    'transcript': None,
                    'duration': call_data.get('duration', 0)
                }
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch transcript: {response.text}")
            
            return {
                'call_id': call_id,
                'status': call_data.get('status', 'unknown'),
                'metadata': call_data.get('metadata', {}),
                'transcript': response.json(),
                'duration': call_data.get('duration', 0)
            }
            
    except Exception as e:
        logger.error(f"❌ Error fetching transcript: {str(e)}")
        raise

def has_sufficient_data(processed_data: Dict[str, Any]) -> bool:
    """Check if we have sufficient data from the call to proceed with matchmaking."""
    required_fields = [
        'skills',
        'years_of_experience',
        'preferred_work_environment',
        'preferred_locations',
        'minimum_salary',
        'work_authorization'
    ]
    return all(processed_data.get(field) for field in required_fields)

async def fetch_and_store_retell_transcript(call_id: str) -> dict:
    """Fetch transcript from Retell and store in memory and Pinecone"""
    try:
        # Check if we've already processed this call
        if call_id in call_statuses and call_statuses[call_id].get('processed'):
            logger.info(f"Call {call_id} already processed, skipping")
            return call_statuses[call_id].get('processed_data', {})

        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 Starting transcript processing for call {call_id}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Get transcript data
        transcript_data = await fetch_retell_transcript(call_id)
        if not transcript_data:
            raise Exception(f"Could not fetch transcript data for {call_id}")
            
        # Determine call status based on transcript and call data
        call_status = "completed"  # default status
        transcript = transcript_data['transcript']
        duration_ms = transcript_data.get('duration', 0)
        
        # Check if call was unanswered
        if transcript_data.get('unanswered', False):
            call_status = "missed"
            logger.info(f"Call {call_id} was not answered")
            # Send missed call email using InteractionAgent
            candidate_email = transcript_data['metadata'].get('email')
            candidate_name = transcript_data['metadata'].get('name', '')  # Get full name from metadata
            candidate_id = transcript_data['metadata'].get('candidate_id')
            if candidate_email and candidate_id:  # Only require email and candidate_id
                await interaction_agent.send_missed_call_email(candidate_email, candidate_name, candidate_id)
        # Check if call was missed (no transcript or very short duration)
        elif not transcript or duration_ms < 1000:  # Less than 1 second
            call_status = "missed"
            logger.info(f"Call {call_id} was missed (no transcript or very short duration)")
            # Send missed call email using InteractionAgent
            candidate_email = transcript_data['metadata'].get('email')
            candidate_name = transcript_data['metadata'].get('name', '')  # Get full name from metadata
            candidate_id = transcript_data['metadata'].get('candidate_id')
            if candidate_email and candidate_id:  # Only require email and candidate_id
                await interaction_agent.send_missed_call_email(candidate_email, candidate_name, candidate_id)
        # Check if call was cut short (short duration or minimal transcript)
        elif duration_ms < 30000 or len(transcript.split()) < 50:  # Less than 30 seconds or less than 50 words
            call_status = "short"
            logger.info(f"Call {call_id} was cut short (short duration or minimal transcript)")
            # Send missed call email for short calls too using InteractionAgent
            candidate_email = transcript_data['metadata'].get('email')
            candidate_name = transcript_data['metadata'].get('name', '')  # Get full name from metadata
            candidate_id = transcript_data['metadata'].get('candidate_id')
            if candidate_email and candidate_id:  # Only require email and candidate_id
                await interaction_agent.send_missed_call_email(candidate_email, candidate_name, candidate_id)
        
        # Process with OpenAI only if we have a completed call
        processed_data = {}
        if call_status == "completed":
            logger.info(f"📝 Processing transcript with OpenAI for call {call_id}")
            
            # Get candidate ID from metadata
            candidate_id = transcript_data['metadata'].get('candidate_id')
            resume_data = ''
            
            if candidate_id:
                # Get resume data from vector store
                try:
                    logger.info(f"🔍 Fetching resume data for candidate {candidate_id}")
                    candidate_query = candidates_index.query(
                        vector=[0] * 1536,
                        filter={"id": candidate_id},
                        top_k=1,
                        include_metadata=True
                    )
                    
                    if candidate_query.matches:
                        resume_data = candidate_query.matches[0].metadata.get('resume_text', '')
                        logger.info(f"✅ Found resume data for candidate {candidate_id}")
                    else:
                        logger.warning(f"⚠️ No resume data found for candidate {candidate_id}")
                except Exception as e:
                    logger.error(f"❌ Error fetching resume data: {str(e)}")
            
            # Combine transcript and resume data
            combined_text = f"""
            Call Transcript:
            {transcript}
            
            Resume Information:
            {resume_data}
            """
            
            processed_data = await process_transcript_with_openai(
                combined_text,
                transcript_data['metadata']  # Pass metadata to get candidate name
            )

            # Check if we have sufficient data for matchmaking
            if has_sufficient_data(processed_data):
                logger.info("✅ Sufficient data gathered from call and resume, proceeding with matchmaking")
                
                try:
                    # Create candidate data for matching
                    candidate_data = {
                        'id': candidate_id,
                        'skills': processed_data.get('skills', []),
                        'years_of_experience': processed_data.get('years_of_experience'),
                        'preferred_work_environment': processed_data.get('preferred_work_environment'),
                        'preferred_locations': processed_data.get('preferred_locations'),
                        'minimum_salary': processed_data.get('minimum_salary'),
                        'work_authorization': processed_data.get('work_authorization'),
                        'email': transcript_data['metadata'].get('email')
                    }
                    
                    # Trigger matchmaking through brain agent
                    match_result = brain_agent.handle_candidate_submission(candidate_data)
                    
                    if match_result.get('status') == 'success':
                        logger.info(f"✅ Successfully matched candidate {candidate_id}")
                        # Add match results to processed data
                        processed_data['match_results'] = match_result.get('match_details')
                    else:
                        logger.warning(f"⚠️ No immediate matches found for candidate {candidate_id}")
                except Exception as e:
                    logger.error(f"❌ Error in matchmaking process: {str(e)}")
                    # Don't raise here as we still want to proceed with email sending
            else:
                logger.info("⚠️ Insufficient data from call and resume for matchmaking")
        else:
            # For missed or short calls, create minimal processed data
            processed_data = {
                'first_name': transcript_data['metadata'].get('name', '').split()[0],
                'call_status': call_status,
                'timestamp': datetime.now().isoformat()
            }
        
        # Store call status in Pinecone if we have a candidate ID
        candidate_id = transcript_data['metadata'].get('candidate_id')
        if candidate_id:
            logger.info(f"💾 Storing call status in Pinecone for candidate {candidate_id}")
            # Serialize processed_data for Pinecone metadata
            pinecone_metadata = {
                'candidate_id': candidate_id,
                'call_id': call_id,
                'processed_at': datetime.now().isoformat(),
                'processed_data': json.dumps(processed_data),  # Serialize the complex object
                'call_status': call_status
            }
            await store_in_pinecone(candidate_id, pinecone_metadata)
        
        # Send email if we have the address and haven't sent it yet
        email = transcript_data['metadata'].get('email')
        if email and call_id in call_statuses and not call_statuses[call_id].get('email_sent'):
            try:
                logger.info(f"\n📧 Email Process Started for call {call_id}")
                logger.info(f"Recipient: {email}")
                logger.info(f"Call Status: {call_status}")
                
                # Check if email was already sent
                logger.info("Checking Pinecone for previous email status...")
                query_response = call_status_index.query(
                    vector=[0] * 1536,
                    filter={"call_id": call_id},
                    top_k=1,
                    include_metadata=True
                )
                
                if not query_response.matches or not query_response.matches[0].metadata.get('email_sent'):
                    logger.info("📧 Sending email via InteractionAgent...")
                    await interaction_agent.send_transcript_summary(email, processed_data, call_status)
                    
                    # Update Pinecone with email sent status
                    try:
                        # Create a vector with one non-zero value
                        vector_values = [0.0] * 1536
                        vector_values[0] = 1.0  # Set first value to 1.0
                        
                        # Serialize processed_data for Pinecone metadata
                        pinecone_metadata = {
                            'status': 'completed',
                            'email_sent': True,
                            'email_sent_at': datetime.utcnow().isoformat(),
                            'processed_data': json.dumps(processed_data),  # Serialize the complex object
                            'call_status': call_status
                        }
                        
                        call_status_index.upsert(vectors=[(
                            call_id,
                            vector_values,
                            pinecone_metadata
                        )])
                        logger.info("✅ Successfully updated Pinecone with email sent status")
                    except Exception as e:
                        logger.error(f"❌ Error updating Pinecone with email status: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        # Don't raise here as the email was sent successfully
                    
                    # Update local status
                    if call_id in call_statuses:
                        call_statuses[call_id]['email_sent'] = True
                        call_statuses[call_id]['email_sent_at'] = datetime.now().isoformat()
                        call_statuses[call_id]['call_status'] = call_status
                    
                    logger.info(f"✅ Email sent successfully to {email}")
                else:
                    logger.info("Email already sent for this call")
            except Exception as e:
                logger.error(f"❌ Error in email sending process: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Update call status
        if call_id in call_statuses:
            call_statuses[call_id].update({
                'processed': True,
                'processed_at': datetime.now().isoformat(),
                'processed_data': processed_data,
                'call_status': call_status
            })
        
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Error processing transcript for call {call_id}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

async def get_retell_call(call_id: str) -> Optional[Dict]:
    """Get call data from Retell API."""
    try:
        logger.info(f"🔄 Fetching call data from Retell for {call_id}")
        async with httpx.AsyncClient(verify=True) as client:
            url = f"{RETELL_API_BASE}/get-call/{call_id}"
            headers = {
                "Authorization": f"Bearer {RETELL_API_KEY}",
                "Content-Type": "application/json"
            }
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error fetching call data: {response.text}")
                return None
    except Exception as e:
        logger.error(f"❌ Error fetching call data: {str(e)}")
        return None

@app.post("/jobs/submit")
async def submit_job(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Submit a job posting by uploading a text file.
    The file content will be processed using OpenAI and stored in Pinecone.
    """
    try:
        print(f"\n=== Processing job submission at {datetime.utcnow().isoformat()} ===")
        
        # Validate file type
        if not file.filename.lower().endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="File must be a text file (.txt or .md)"
            )
        
        # Read the file content
        content = await file.read()
        try:
            raw_text = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be UTF-8 encoded text"
            )
        
        # Generate a unique job ID
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        print(f"Generated job ID: {job_id}")
        
        # Update job status
        job_statuses[job_id] = {
            "status": JobStatus.PROCESSING,
            "progress": 0,
            "message": "Starting job analysis"
        }
        
        try:
            # Process job text with OpenAI
            print("Processing with OpenAI...")
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": JOB_ANALYSIS_PROMPT},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the OpenAI response
            try:
                processed_data = json.loads(response.choices[0].message.content)
                print("Successfully processed job text with OpenAI")
            except json.JSONDecodeError as e:
                print(f"Error parsing OpenAI response: {e}")
                print(f"Raw response: {response.choices[0].message.content}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse job analysis response"
                )
            
            # Update job status
            job_statuses[job_id] = {
                "status": JobStatus.PROCESSING,
                "progress": 50,
                "message": "Storing job in database"
            }
            
            # Store in vector database
            vector_store = VectorStore(init_openai=True)
            store_result = vector_store.store_job(job_id, processed_data)
            
            if store_result.get("status") == "error":
                raise Exception(store_result.get("message", "Unknown error storing job"))
            
            # Update final status
            job_statuses[job_id] = {
                "status": JobStatus.COMPLETED,
                "progress": 100,
                "message": "Job processed and stored successfully",
                "job_id": job_id,
                "processed_data": processed_data
            }
            
            return {
                "status": "success",
                "message": "Job processed and stored successfully",
                "job_id": job_id,
                "processed_data": processed_data
            }
            
        except Exception as e:
            print(f"Error processing job submission: {str(e)}")
            job_statuses[job_id] = {
                "status": JobStatus.FAILED,
                "progress": 0,
                "message": f"Error: {str(e)}"
            }
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process job submission: {str(e)}"
            )
            
    except Exception as e:
        print(f"Error in job submission endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing job submission: {str(e)}"
        )
    finally:
        print("=== Job submission processing complete ===\n")

class InboundEmailData(BaseModel):
    headers: Dict[str, str]
    text: str
    html: Optional[str]
    from_email: str
    subject: str
    to: str
    envelope: Dict[str, Any]
    attachments: Optional[List[Dict[str, Any]]] = []
    
async def process_candidate_email(email_data: InboundEmailData) -> Dict[str, Any]:
    """Process an incoming email from a candidate."""
    try:
        logger.info(f"🔍 Processing email from: {email_data.from_email}")
        
        # Extract candidate email from the 'from' field
        candidate_email = email_data.from_email
        
        # Search for candidate in Pinecone by email
        logger.info("🔎 Searching for candidate in database...")
        candidate_query = {
            'filter': {
                'email': candidate_email
            }
        }
        
        results = candidates_index.query(
            vector=[0] * 1536,  # Dummy vector for query
            filter=candidate_query['filter'],
            top_k=1,
            include_metadata=True
        )
        
        if not results.matches:
            logger.error(f"❓ No candidate found for email: {candidate_email}")
            return {
                'status': 'error',
                'message': 'Candidate not found'
            }
            
        candidate_id = results.matches[0].id
        candidate_metadata = results.matches[0].metadata
        logger.info(f"✅ Found candidate: {candidate_metadata.get('name', 'Unknown')}")
        
        # Process the email content with OpenAI for internal analysis
        logger.info("🤖 Analyzing email content with AI...")
        analysis_prompt = f"""Analyze this email from a candidate and help understand their question or concern:

        Email Subject: {email_data.subject}
        Email Content: {email_data.text}
        
        Please provide:
        1. The main question or concern
        2. Any specific job roles mentioned
        3. Any new preferences or requirements mentioned
        4. Suggested response approach
        
        Format the response as JSON with these keys:
        {{
            "main_question": string,
            "mentioned_jobs": list[string],
            "new_preferences": dict,
            "response_approach": string,
            "requires_human_review": boolean
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping to analyze candidate emails."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={ "type": "json_object" }
        )
        
        analysis = json.loads(response.choices[0].message.content)
        logger.info(f"📊 Analysis complete:")
        logger.info(f"❓ Main question: {analysis['main_question']}")
        logger.info(f"💼 Jobs mentioned: {', '.join(analysis['mentioned_jobs']) if analysis['mentioned_jobs'] else 'None'}")
        logger.info(f"👀 Requires human review: {'Yes' if analysis['requires_human_review'] else 'No'}")
        
        # Update candidate profile with new preferences
        if analysis['new_preferences']:
            logger.info("📝 Updating candidate preferences...")
            updated_metadata = candidate_metadata.copy()
            updated_metadata['preferences'] = {
                **updated_metadata.get('preferences', {}),
                **analysis['new_preferences']
            }
            updated_metadata['last_interaction'] = datetime.now().isoformat()
            updated_metadata['interaction_history'] = updated_metadata.get('interaction_history', []) + [{
                'type': 'email_reply',
                'timestamp': datetime.now().isoformat(),
                'content': email_data.text,
                'analysis': analysis
            }]
            
            # Store vector of email content for future reference
            logger.info("🔤 Generating email content embedding...")
            email_embedding = await get_embedding(email_data.text)
            
            # Update candidate in Pinecone
            logger.info("💾 Updating candidate profile in database...")
            candidates_index.upsert(vectors=[(
                candidate_id,
                email_embedding,
                updated_metadata
            )])
        
        # Generate and send response
        if not analysis['requires_human_review']:
            logger.info("✍️ Generating AI response...")
            # Get candidate's first name
            candidate_name = candidate_metadata.get('name', '').split()[0]
            
            response_prompt = f"""You are Anita, an AI Career Co-Pilot having a friendly email conversation with {candidate_name}. Write a natural, helpful response to their email.

            Context:
            - Their email: {email_data.text}
            - Their question/concern: {analysis['main_question']}
            - Jobs they mentioned: {', '.join(analysis['mentioned_jobs']) if analysis['mentioned_jobs'] else 'None'}
            - Previous interactions: {json.dumps(updated_metadata.get('interaction_history', [])[-3:], indent=2)}

            Guidelines:
            1. Write in a warm, professional tone
            2. Address them by first name
            3. Directly answer their specific question/concern
            4. If they mentioned specific jobs, provide relevant details
            5. Ask follow-up questions to better understand their needs
            6. Keep the response concise but helpful
            7. Sign as "Anita" without mentioning you're an AI
            8. Don't use any JSON formatting - write a natural email

            Write your response now:"""
            
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are Anita, writing a friendly and professional email response to a candidate."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            reply_content = response.choices[0].message.content
            logger.info("📧 Sending email response...")
            
            # Send email reply
            await interaction_agent.send_email(
                to_email=candidate_email,
                subject=f"Re: {email_data.subject}",
                content=reply_content
            )
            logger.info("✅ Email sent successfully!")
        else:
            # Send Slack notification for human review
            logger.info("👥 Preparing notification for human review...")
            candidate_name = candidate_metadata.get('name', 'Unknown')
            slack_message = f"""🔍 *Human Review Needed for Candidate Email*

*From:* {candidate_name} ({candidate_email})
*Subject:* {email_data.subject}
*Question/Concern:* {analysis['main_question']}

*Email Content:*
{email_data.text}

*Analysis:*
• Jobs Mentioned: {', '.join(analysis['mentioned_jobs']) if analysis['mentioned_jobs'] else 'None'}
• New Preferences: {json.dumps(analysis['new_preferences'], indent=2) if analysis['new_preferences'] else 'None'}
• Suggested Approach: {analysis['response_approach']}

Please review and respond to this email as soon as possible."""
            
            await send_slack_notification(slack_message)
            logger.info("✅ Slack notification sent!")
            
        return {
            'status': 'success',
            'candidate_id': candidate_id,
            'requires_human_review': analysis['requires_human_review'],
            'analysis': analysis
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing candidate email: {str(e)}")
        logger.error(f"💥 Error type: {type(e)}")
        logger.error(f"📜 Stack trace: {traceback.format_exc()}")
        return {
            'status': 'error',
            'message': str(e)
        }

@app.post("/email/webhook")
async def handle_sendgrid_webhook(request: Request):
    """Handle incoming emails from SendGrid's Inbound Parse webhook."""
    logger.info("📨 Received webhook from SendGrid")
    
    try:
        # Log the raw request data
        body = await request.body()
        form = await request.form()
        logger.info(f"📝 Received form data keys: {form.keys()}")
        
        # Log headers for debugging
        headers = dict(request.headers)
        logger.info(f"🔍 Request headers: {headers}")
        
        # Extract email data
        email_data = {
            "to": form.get("to", ""),
            "from": form.get("from", ""),
            "subject": form.get("subject", ""),
            "text": form.get("text", ""),
            "html": form.get("html", ""),
        }
        logger.info(f"📧 Processed email data: {email_data}")
        
        # Notify Slack for debugging
        await send_slack_notification(
            f"*New Email Received*\n"
            f"From: {email_data['from']}\n"
            f"Subject: {email_data['subject']}\n"
            f"Content: {email_data['text'][:200]}..."
        )
        
        return {"status": "success", "message": "Email processed successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}")
        logger.error(f"💥 Error type: {type(e)}")
        logger.error(f"📜 Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using OpenAI's API."""
    try:
        # Ensure text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
            
        # Truncate text if too long (OpenAI has token limits)
        max_tokens = 8000  # Conservative limit
        text = text[:max_tokens]
        
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        return response.data[0].embedding
        
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        # Return a zero vector as fallback
        return [0.0] * 1536

@app.get("/test-slack")
async def test_slack_notification():
    """Test endpoint to verify Slack notifications are working."""
    message = (
        "🧪 *Test Notification*\n\n"
        "This is a test message from Recruitcha's email processing system.\n"
        "• Timestamp: {}\n"
        "• Environment: Development\n"
        "• Status: Testing"
    ).format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    success = await send_slack_notification(message)
    
    if success:
        return {"status": "success", "message": "Slack notification sent successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send Slack notification")

@app.get("/calls/initiate/{candidate_id}")
async def initiate_call(candidate_id: str):
    """Initiate a Retell AI call for a candidate."""
    try:
        # Get candidate's phone number from vector store
        try:
            # Query using metadata filter instead of vector ID
            candidate_query = candidates_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata-only query
                filter={"candidate_id": candidate_id},
                top_k=1,
                include_metadata=True
            )
            
            if not candidate_query.matches:
                raise HTTPException(
                    status_code=404,
                    detail=f"Candidate {candidate_id} not found"
                )
            
            candidate_metadata = candidate_query.matches[0].metadata
            phone_number = candidate_metadata.get('phone_number')
            
            if not phone_number:
                raise HTTPException(
                    status_code=400,
                    detail=f"No phone number found for candidate {candidate_id}"
                )
            
            # Prepare call data
            call_data = {
                "from_number": RETELL_FROM_NUMBER,
                "to_number": phone_number,
                "metadata": {
                    "candidate_id": candidate_id,
                    "name": candidate_metadata.get('name', ''),
                    "email": candidate_metadata.get('email', '')
                }
            }
            
            # Make request to Retell API
            headers = {
                "Authorization": f"Bearer {RETELL_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{RETELL_API_BASE}/create-phone-call",
                    json=call_data,
                    headers=headers
                )
                
                if response.status_code != 201:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to initiate call: {response.text}"
                    )
                
                call_response = response.json()
                logger.info(f"✅ Successfully initiated call for candidate {candidate_id}")
                
                return {
                    "status": "success",
                    "message": "Call initiated successfully",
                    "call_id": call_response.get('call_id')
                }
                
        except Exception as e:
            logger.error(f"❌ Error initiating call: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initiate call: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"❌ Error in initiate_call endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

async def fetch_retell_call_data(call_id: str) -> Dict[str, Any]:
    """Fetch call data from Retell API."""
    try:
        logger.info(f"🔄 Fetching call data for {call_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{RETELL_API_BASE}/get-call/{call_id}",
                headers={"Authorization": f"Bearer {RETELL_API_KEY}"}
            )
            
            if response.status_code == 404:
                logger.warning(f"⚠️ Call {call_id} not found")
                return {}
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch call data: {response.text}")
            
            return response.json()
            
    except Exception as e:
        logger.error(f"❌ Error fetching call data: {str(e)}")
        raise

app = app