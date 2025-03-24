from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
from openai import OpenAI
import io
import tempfile
import requests
from typing import Annotated
import aiohttp
import traceback
import asyncio
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, VectorType
import phonenumbers
from retell import Retell
import time
import re
from urllib.parse import urlparse
import uuid
import logging
import logging.handlers

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Changed to WARNING to show only important messages
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Only use console logging for Vercel compatibility
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

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
JOB_ANALYSIS_PROMPT = """
Please analyze the following job posting text and extract detailed information into a structured format.
The text may include a transcript of a conversation about the role and/or a formal job description.

Return the information as a JSON object with the following fields:
{{
    "company_name": string,
    "company_website": string,
    "paraform_url": string,
    "company_stage": string,
    "funding_details": {{
        "most_recent_round": string,
        "total_funding": string,
        "key_investors": [string]
    }},
    "team_size": string,
    "founding_year": string,
    "company_mission": string,
    "target_market": [string],
    "industry_vertical": string,
    "company_vision": string,
    "company_growth_story": string,
    "company_culture": {{
        "work_environment": string,
        "decision_making": string,
        "collaboration_style": string,
        "risk_tolerance": string,
        "values": string
    }},
    "job_title": string,
    "positions_available": string,
    "hiring_urgency": string,
    "seniority_level": string,
    "work_arrangement": string,
    "location": {{
        "city": string,
        "state": string,
        "office_details": string
    }},
    "visa_sponsorship": string,
    "compensation": {{
        "base_salary_range": string,
        "equity_details": string,
        "total_comp_range": string
    }},
    "reporting_structure": string,
    "team_composition": string,
    "role_category": string,
    "tech_stack": {{
        "must_haves": [string],
        "nice_to_haves": [string],
        "tools_and_frameworks": [string]
    }},
    "experience_requirements": {{
        "minimum_years": string,
        "level": string,
        "domain_expertise": [string],
        "specific_skills": string
    }},
    "education_requirements": {{
        "minimum": string,
        "preferred": string,
        "notes": string
    }},
    "key_responsibilities": [string],
    "ideal_candidate_profile": string,
    "interview_process": {{
        "stages": [string],
        "work_trial_details": string,
        "timeline": string
    }},
    "deal_breakers": [string],
    "growth_opportunities": string
}}

For any fields where information is not explicitly mentioned in the text, use "Not specified" for string fields and [] for array fields.

Be particularly careful to:
1. Extract the company website URL if mentioned
2. Extract the Paraform job posting URL if present
3. Capture compensation details including base salary, equity, and total comp ranges
4. Extract technical requirements and must-have skills
5. Identify work arrangement and location details
6. Note interview process specifics
7. Include company stage and funding information

Text to analyze:
{raw_text}
"""

# Initialize Pinecone
logger.info("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create call-statuses index if it doesn't exist
existing_indexes = [index.name for index in pc.list_indexes()]
if "call-statuses" not in existing_indexes:
    logger.info("Creating call-statuses index...")
    pc.create_index(
        name="call-statuses",
        dimension=1536,  # Same dimension as other indexes
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        )
    )
    logger.info("Successfully created call-statuses index")
else:
    logger.info("call-statuses index already exists")

job_index = pc.Index("job-details")
call_status_index = pc.Index("call-statuses")  # New index for call statuses

app = FastAPI(
    title="Anita AI Recruitment API",
    description="API for AI-driven recruitment with enhanced candidate-job matching",
    version="2.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

interaction_agent = InteractionAgent()
brain_agent = BrainAgent()
vector_store = VectorStore()

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
    REGISTERED = "registered"
    ONGOING = "ongoing"
    ENDED = "ended"
    ERROR_RETELL = "error_retell"
    ERROR_UNKNOWN = "error_unknown"
    ERROR_USER_NOT_JOINED = "error_user_not_joined"
    REGISTERED_CALL_TIMEOUT = "registered_call_timeout"

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

class RetellWebhookPayload(BaseModel):
    call_id: str
    call_status: RetellCallStatus
    metadata: Dict[str, Any]
    transcript: Optional[str] = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
                for page in pdf_reader.pages:
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(text)
                    except Exception as page_error:
                        print(f"Error extracting text from page: {str(page_error)}")
                        continue
                
                if not text_content:
                    raise HTTPException(
                        status_code=400, 
                        detail="No text content could be extracted from the PDF. The file might be scanned or image-based."
                    )
                
                # Combine all text
                combined_text = "\n\n".join(text_content)
                
                return {
                    "text": combined_text,
                    "filename": file.filename,
                    "page_count": len(pdf_reader.pages),
                    "text_pages": len(text_content)
                }
                    
            except Exception as pdf_error:
                print(f"Error reading PDF: {str(pdf_error)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"The PDF file appears to be corrupted or in an unsupported format. Please ensure you're uploading a valid PDF file. Error: {str(pdf_error)}"
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
            detail=f"Failed to process PDF file: {str(e)}"
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
        
        # Process the PDF file
        try:
            pdf_result = await process_pdf_to_text(resume)
            resume_text = pdf_result["text"]
            print("Successfully extracted text from resume")
        except HTTPException as pdf_error:
            raise HTTPException(
                status_code=pdf_error.status_code,
                detail=f"Failed to process resume: {pdf_error.detail}"
            )

        # Create the candidate profile
        profile = {
            "id": candidate_id,
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "linkedin": linkedin,
            "resume_text": resume_text
        }
        print("Created candidate profile")

        # Store candidate in vector database
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

async def process_transcript_with_openai(transcript: str) -> Dict[str, Any]:
    """Process transcript with OpenAI to extract structured information."""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": """You are an expert at analyzing job interview transcripts. Extract the following information in a structured format:
                - key_points: List of key points discussed in the conversation
                - experience_highlights: List of notable experience or skills mentioned
                - next_steps: A brief summary of next steps or action items
                - candidate_name: The candidate's name if mentioned
                - contact_information: Any contact details mentioned (email, phone, LinkedIn)
                - years_of_experience: Years of experience if mentioned
                - skills: List of skills mentioned
                - preferred_work_environment: Preferred work environment if mentioned
                - preferred_locations: Preferred locations if mentioned
                - minimum_salary: Salary expectations if mentioned
                - work_authorization: Work authorization status if mentioned

                Format the response as a valid JSON object with these exact keys."""},
                {"role": "user", "content": f"Please analyze this interview transcript and extract the key information:\n\n{transcript}"}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"Error processing transcript with OpenAI: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
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
        processed_data = await process_transcript_with_openai(transcript_data.transcript)
        
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

@app.post("/test-email")
async def test_email():
    # Test data for Andrew
    job_match = {
        'email': 'arschuessler90@gmail.com',
        'phone_number': '+18476094515',
        'title': 'Senior Backend Engineer',
        'company': 'Hedra'
    }
    
    result = interaction_agent.contact_candidate(job_match)
    return result

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
        logger.info("\n=== Syncing call statuses from Pinecone ===")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
        
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
                        logger.info(f"Synced call {call_id} with status: {metadata.get('status', 'unknown')}")
                    except Exception as match_error:
                        logger.error(f"Error processing match for call {match.id}: {str(match_error)}")
                        logger.error(f"Match data: {match}")
                        continue
                
                logger.info(f"Successfully synced {len(call_statuses)} call statuses from Pinecone")
            else:
                logger.info("No call statuses found in Pinecone")
                
        except Exception as query_error:
            logger.error(f"Error querying Pinecone: {str(query_error)}")
            logger.error(f"Error type: {type(query_error)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise
            
    except Exception as e:
        logger.error(f"Critical error in sync_call_statuses: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
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
        
        if 'status' not in status_data:
            raise ValueError("status_data must contain a 'status' field")
        
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
        
        # Create RetellCallData object for processing
        call_data = RetellCallData(
            call_id=call_id,
            candidate_id=status_data.get('candidate_id'),
            email=email
        )
        
        # Process the transcript and send email
        await fetch_and_store_retell_transcript(call_data)
        
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
    """Initialize services and background tasks on startup"""
    try:
        await sync_call_statuses()
        num_removed = clean_test_calls()
        if num_removed > 0:
            logger.info(f"Removed {num_removed} test calls")
        
        # Start the cleanup task
        cleanup_task = asyncio.create_task(cleanup_completed_calls())
        app.state.cleanup_task = cleanup_task
        logger.info("Started cleanup_completed_calls background task")
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on shutdown"""
    try:
        if hasattr(app.state, 'cleanup_task'):
            app.state.cleanup_task.cancel()
            try:
                await app.state.cleanup_task
            except asyncio.CancelledError:
                pass
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

        # Process resume with OpenAI to get structured data including current company
        pdf_result = await process_pdf_to_text(resume)
        resume_text = pdf_result["text"]
        resume_data = {
            "text": resume_text,
            "candidate_id": candidate_id
        }
        processed_resume = await process_resume_text(resume_data)
        current_company = processed_resume.get("structured_data", {}).get("current_company", "")
        print(f"Extracted current company: {current_company}")

        # Create a summary of key information from the resume
        structured_data = processed_resume.get("structured_data", {})
        skills = structured_data.get('skills', [])
        # Handle skills array properly
        top_skills = []
        if isinstance(skills, list):
            top_skills = skills[:5] if len(skills) > 0 else []
        skills_text = ', '.join(top_skills) if top_skills else 'Not specified'
        
        resume_summary = f"""
        Current Role: {structured_data.get('current_role', 'Not specified')}
        Current Company: {structured_data.get('current_company', 'Not specified')}
        Years of Experience: {len(structured_data.get('experience', []))} years
        Key Skills: {skills_text}
        Education: {structured_data.get('education', ['Not specified'])[0] if structured_data.get('education') else 'Not specified'}
        Summary: {structured_data.get('summary', 'Not specified')}
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
                "resume_summary": resume_summary  # Add the summarized resume information
            }
        }

        print(f"\nMaking call with payload: {json.dumps(retell_payload, indent=2)}")

        # Make the call
        async with httpx.AsyncClient() as client:
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
                "current_company": current_company
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

@app.post("/delete-knowledge-base/{knowledge_base_id}")
async def delete_knowledge_base(knowledge_base_id: str):
    """Delete a knowledge base from Retell AI."""
    try:
        success = await delete_retell_knowledge_base(knowledge_base_id)
        if success:
            return {
                "status": "success",
                "message": f"Successfully deleted knowledge base {knowledge_base_id}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete knowledge base {knowledge_base_id}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def check_call_status(call_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Check the status of a call with Retell AI."""
    try:
        async with httpx.AsyncClient() as client:
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

@app.post("/webhook/retell")
async def retell_webhook(request: Request):
    """Handle webhook notifications from Retell AI."""
    try:
        print("\n=== Received Retell Webhook ===")
        # Get raw data first
        raw_data = await request.json()
        print(f"Raw webhook data: {json.dumps(raw_data, indent=2)}")
        
        # Extract required fields with fallbacks
        call_id = raw_data.get('call_id')
        if not call_id:
            print("Missing call_id in webhook data")
            return {"status": "error", "message": "Missing call_id"}
        
        # Convert webhook status to our enum if possible
        webhook_status = raw_data.get('call_status', 'unknown')
        try:
            call_status = RetellCallStatus(webhook_status)
        except ValueError:
            print(f"Invalid call status: {webhook_status}")
            call_status = RetellCallStatus.ERROR_UNKNOWN
        
        # Get metadata with fallback
        metadata = raw_data.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
            
        # Prepare status data
        status_data = {
            "status": call_status,
            "webhook_received_at": datetime.utcnow().isoformat(),
            "raw_status": webhook_status,
            "metadata": metadata
        }
        
        # Add transcript if present
        if 'transcript' in raw_data:
            status_data['transcript'] = raw_data['transcript']
            status_data['has_transcript'] = True
        
        # Add any additional fields from the webhook
        for key, value in raw_data.items():
            if key not in ['call_id', 'call_status', 'metadata', 'transcript']:
                status_data[key] = value
        
        try:
            # Update call status
            await update_call_status(call_id, status_data)
            print(f"Successfully processed webhook for call {call_id}")

            # If the call has ended, process transcript and send email
            if call_status == RetellCallStatus.ENDED:
                print(f"Call {call_id} has ended, processing transcript...")
                
                # Get the call data from our status storage
                stored_status = call_statuses.get(call_id, {})
                
                # Create call data object for processing
                call_data = RetellCallData(
                    call_id=call_id,
                    candidate_id=stored_status.get('candidate_id') or metadata.get('candidate_id'),
                    email=stored_status.get('candidate_email') or metadata.get('email')
                )
                
                # Process the transcript and send email
                try:
                    processed_data = await fetch_and_store_retell_transcript(call_data)
                    print(f"Successfully processed transcript for call {call_id}")
                    
                    # Send email if we have the candidate's email
                    candidate_email = stored_status.get('candidate_email') or metadata.get('email')
                    if candidate_email:
                        print(f"Sending email to {candidate_email}...")
                        interaction_agent = InteractionAgent()
                        email_result = interaction_agent.send_transcript_summary(
                            candidate_email,
                            processed_data
                        )
                        
                        if email_result['status'] == 'success':
                            print(f"Successfully sent email to {candidate_email}")
                            status_data['email_sent'] = True
                            status_data['email_sent_at'] = datetime.utcnow().isoformat()
                            status_data['email_status'] = email_result
                            await update_call_status(call_id, status_data)
                        else:
                            print(f"Failed to send email: {email_result.get('error')}")
                    
                except Exception as process_error:
                    print(f"Error processing transcript: {str(process_error)}")
                    print(f"Error type: {type(process_error)}")
                    print(f"Error traceback: {traceback.format_exc()}")
            
            return {"status": "success", "call_id": call_id, "processed_status": call_status}
            
        except Exception as update_error:
            print(f"Error updating call status: {str(update_error)}")
            print(f"Error type: {type(update_error)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return {"status": "error", "message": str(update_error)}
        
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def clean_test_calls():
    """Remove test calls from call_statuses dictionary."""
    global call_statuses
    test_calls = [call_id for call_id in call_statuses.keys() if call_id.startswith('test_') or call_id.startswith('call_test_')]
    for call_id in test_calls:
        del call_statuses[call_id]
    return len(test_calls)

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
    """Background task to clean up completed calls and process their transcripts."""
    print("\n=== Starting Call Cleanup Task ===")
    while True:
        try:
            # Get all unprocessed calls
            unprocessed_calls = {
                call_id: status_data
                for call_id, status_data in call_statuses.items()
                if not status_data.get('processed_by_system')
            }
            
            if not unprocessed_calls:
                print("No new calls to process")
            else:
                print(f"Found {len(unprocessed_calls)} unprocessed calls")
                
                for call_id, status_data in unprocessed_calls.items():
                    try:
                        print(f"\nProcessing call {call_id}...")
                        
                        # Create call data object for processing
                        call_data_obj = RetellCallData(
                            call_id=call_id,
                            candidate_id=status_data.get('candidate_id'),
                            email=status_data.get('candidate_email')
                        )
                        
                        try:
                            # Try to fetch and process transcript
                            processed_data = await fetch_and_store_retell_transcript(call_data_obj)
                            print(f"Successfully processed transcript for call {call_id}")
                            
                            # Send email if we have the candidate's email
                            if status_data.get('candidate_email'):
                                print(f"Sending email to {status_data['candidate_email']}...")
                                interaction_agent = InteractionAgent()
                                email_result = interaction_agent.send_transcript_summary(
                                    status_data['candidate_email'],
                                    processed_data
                                )
                                
                                if email_result['status'] == 'success':
                                    print(f"Successfully sent email to {status_data['candidate_email']}")
                                    status_data['email_sent'] = True
                                    status_data['email_sent_at'] = datetime.utcnow().isoformat()
                                    status_data['email_status'] = email_result
                                else:
                                    print(f"Failed to send email: {email_result.get('error')}")
                                    
                        except HTTPException as he:
                            if he.status_code == 404 and "No transcript found" in str(he.detail):
                                print(f"No transcript available for call {call_id} - call may have ended prematurely")
                                status_data['transcript_status'] = "No transcript available - call may have ended prematurely"
                            else:
                                print(f"Error processing call {call_id}: {str(he)}")
                                status_data['error'] = str(he)
                        except Exception as e:
                            print(f"Error processing call {call_id}: {str(e)}")
                            print(f"Error type: {type(e)}")
                            print(f"Error traceback: {traceback.format_exc()}")
                            status_data['error'] = str(e)
                        
                        # Mark as processed regardless of outcome
                        status_data['processed_by_system'] = True
                        status_data['processed_at'] = datetime.utcnow().isoformat()
                        await update_call_status(call_id, status_data)
                        
                    except Exception as e:
                        print(f"ERROR: Error processing call {call_id}: {str(e)}")
                        print(f"ERROR: Error traceback: {traceback.format_exc()}")
            
            # Sleep for 30 seconds before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"ERROR in cleanup task: {str(e)}")
            print(f"ERROR traceback: {traceback.format_exc()}")
            await asyncio.sleep(30)  # Sleep even on error to prevent rapid retries

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

async def fetch_retell_transcript(call_id: str) -> Optional[Dict]:
    """Fetch transcript from Retell API"""
    try:
        headers = {
            "Authorization": f"Bearer {RETELL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        url = f"{RETELL_API_BASE}/get-call/{call_id}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 404:
                    logger.error(f"Call {call_id} not found")
                    raise HTTPException(
                        status_code=404,
                        detail={
                            'error': 'Call not found',
                            'message': f'No call found with ID: {call_id}',
                            'action_required': 'Please verify the call ID'
                        }
                    )
                elif response.status_code == 401:
                    logger.error("Authentication failed - invalid API key")
                    raise HTTPException(
                        status_code=401,
                        detail={
                            'error': 'Authentication failed',
                            'message': 'Invalid API key',
                            'action_required': 'Please check your Retell API key'
                        }
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # Log only essential information
                logger.info(f"Successfully fetched transcript for call {call_id}")
                logger.debug(f"Call data: {data.get('metadata', {})}")
                
                return data
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching transcript for call {call_id}: {str(e)}")
                if not isinstance(e, HTTPException):
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail={
                            'error': 'Retell API error',
                            'message': str(e),
                            'action_required': 'Please try again or contact support'
                        }
                    )
                raise
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching transcript for call {call_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Internal server error',
                'message': str(e),
                'action_required': 'Please try again or contact support'
            }
        )

@app.get("/calls/recent", response_model=Dict[str, Any])
async def get_recent_calls():
    """Get recent Retell AI calls from Pinecone."""
    try:
        print("\n=== Fetching Recent Calls ===")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        
        # Query Pinecone for recent calls
        query_response = call_status_index.query(
            vector=[0] * 1536,  # Dummy vector since we're just getting metadata
            top_k=10,  # Get 10 most recent calls
            include_metadata=True
        )
        
        calls = []
        if query_response.matches:
            for match in query_response.matches:
                call_data = match.metadata
                call_data['call_id'] = match.id
                calls.append(call_data)
            
            print(f"Found {len(calls)} recent calls")
            return {
                "status": "success",
                "calls": calls,
                "total": len(calls)
            }
        else:
            print("No calls found in Pinecone")
            return {
                "status": "success",
                "calls": [],
                "total": 0
            }
            
    except Exception as e:
        print(f"Error fetching recent calls: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch recent calls: {str(e)}"
        )

@app.post("/calls/{call_id}/end")
async def end_call(call_id: str):
    """Manually mark a call as ended."""
    try:
        print(f"\n=== Marking call {call_id} as ended ===")
        
        if call_id not in call_statuses:
            raise HTTPException(
                status_code=404,
                detail=f"Call {call_id} not found"
            )
            
        # Update call status
        status_data = call_statuses[call_id]
        status_data['status'] = RetellCallStatus.ENDED
        status_data['last_updated'] = datetime.utcnow().isoformat()
        
        await update_call_status(call_id, status_data)
        
        return {
            "status": "success",
            "message": f"Call {call_id} marked as ended",
            "call_status": status_data
        }
        
    except Exception as e:
        print(f"Error marking call as ended: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mark call as ended: {str(e)}"
        )

@app.get("/candidates/{candidate_id}/complete-profile")
async def get_complete_candidate_profile(candidate_id: str):
    """Get a candidate's complete profile including all information, call transcripts, and processed data."""
    try:
        # Initialize required components
        vector_store = VectorStore(init_openai=True)
        brain_agent = BrainAgent()
        
        # Get basic profile from vector store
        vector_profile = vector_store.get_candidate_profile(candidate_id)
        if vector_profile["status"] == "error":
            raise HTTPException(
                status_code=404,
                detail=vector_profile["message"]
            )
        
        # Get additional data from brain agent
        brain_profile = brain_agent.candidate_profiles.get(candidate_id, {})
        
        # If brain_profile is empty, try to fetch call data from Retell AI
        if not brain_profile:
            # Query call_statuses to find the call_id for this candidate
            call_id = None
            for call_id, call_data in call_statuses.items():
                if call_data.get('candidate_id') == candidate_id:
                    call_id = call_id
                    break
            
            if call_id:
                try:
                    # Fetch call data from Retell AI
                    retell_data = await fetch_retell_transcript(call_id)
                    if retell_data and 'transcript' in retell_data:
                        # Process the transcript
                        processed_data = await process_transcript_with_openai(retell_data['transcript'])
                        brain_profile = {
                            'transcript': retell_data['transcript'],
                            'processed_transcript': processed_data
                        }
                        # Optionally update brain_agent's candidate_profiles
                        brain_agent.candidate_profiles[candidate_id] = brain_profile
                except Exception as e:
                    print(f"Error fetching call data for candidate {candidate_id}: {str(e)}")
        
        # Combine all data
        complete_profile = {
            "status": "success",
            "candidate_id": candidate_id,
            "basic_info": vector_profile.get("basic_info", {}),
            "resume_text": vector_profile.get("resume_text", ""),
            "processed_resume": vector_profile.get("processed_data", {}),
            "call_data": {
                "transcript": brain_profile.get("transcript"),
                "processed_transcript": brain_profile.get("processed_transcript"),
                "screening_result": brain_profile.get("screening_result"),
                "match_result": brain_profile.get("match_result"),
                "dealbreakers": brain_profile.get("dealbreakers"),
                "match_reason": brain_profile.get("match_reason")
            },
            "state": brain_agent.state.get(candidate_id, "unknown"),
            "total_chunks": vector_profile.get("total_chunks", 0)
        }
        
        return complete_profile
        
    except Exception as e:
        print(f"Error retrieving complete candidate profile: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/calls/{call_id}/process-transcript")
async def process_call_transcript(call_id: str):
    """Manually process a call's transcript."""
    try:
        # Get call data from Retell AI
        retell_data = await fetch_retell_transcript(call_id)
        if not retell_data:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for call {call_id}"
            )
            
        # Create call data object
        call_data = {
            'call_id': call_id,
            'candidate_id': retell_data.get('metadata', {}).get('candidate_id'),
            'candidate_email': retell_data.get('metadata', {}).get('email'),
            'candidate_name': retell_data.get('metadata', {}).get('name')
        }
        
        # Process the transcript
        processed_data = await fetch_and_store_retell_transcript(call_data)
        
        return {
            "status": "success",
            "message": "Transcript processed successfully",
            "data": processed_data
        }
        
    except Exception as e:
        print(f"Error processing call transcript: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/jobs/match-candidates", response_model=MatchResponse)
async def match_candidates_to_job(request: JobMatchRequest):
    """Match candidates to a job based on job ID."""
    try:
        vector_store = VectorStore(init_openai=True)
        result = vector_store.match_candidates_to_job(request.job_id, request.top_k)
        
        return {
            "status": "success",
            "matches": result['matches'],
            "total_matches": len(result['matches'])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error matching candidates to job: {str(e)}"
        )

app = app