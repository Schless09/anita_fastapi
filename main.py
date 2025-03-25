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
from openai import AsyncOpenAI

# Ensure logs directory exists with proper permissions
os.makedirs("logs", exist_ok=True)
os.chmod("logs", 0o777)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only use console logging for Vercel
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

async def process_transcript_with_openai(transcript: str) -> dict:
    """Process a transcript with OpenAI to extract structured information."""
    try:
        # Create OpenAI client
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Format prompt for analysis
        prompt = f"""
        Please analyze this interview transcript and extract the following information:
        1. Key skills and technologies mentioned
        2. Years of experience
        3. Current role and responsibilities
        4. Notable achievements
        5. Education and certifications
        6. Preferred work environment
        7. Career goals
        8. Salary expectations (if mentioned)
        9. Overall impression
        
        Transcript:
        {transcript}
        
        Please format the response as a JSON object with these fields.
        """
        
        # Call OpenAI API
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes interview transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse OpenAI response as JSON")
            return {"error": "Failed to parse response"}
            
    except Exception as e:
        logger.error(f"Error processing transcript with OpenAI: {str(e)}")
        return {"error": str(e)}

async def store_in_pinecone(candidate_id: str, processed_data: dict):
    """Store processed transcript data in Pinecone"""
    try:
        vector_store = VectorStore(init_openai=True)
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
async def retell_webhook(request: Request):
    """Handle Retell webhook events"""
    try:
        logger.info("\n" + "="*50)
        logger.info("📞 Received Retell webhook")
        logger.info("="*50)

        data = await request.json()
        event_type = data.get("event")
        call_data = data.get("call", {})
        call_id = call_data.get("call_id")

        if not call_id:
            logger.error("❌ Missing call_id in webhook data")
            return {"status": "error", "message": "Missing call_id"}, 400

        logger.info(f"📞 Processing {event_type} event for call: {call_id}")
        logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
        logger.info(f"📋 Call Status: {call_data.get('call_status', 'unknown')}")

        # Log the full webhook data to Pinecone
        log_webhook(call_id, data)
        logger.info("📝 Logged webhook data to Pinecone")

        # Handle different event types
        if event_type == "call_started":
            logger.info(f"🟢 Call started: {call_id}")
            # Update call status to in_progress
            await update_call_status(call_id, {
                "status": "in_progress",
                "start_timestamp": datetime.now().isoformat(),
                "metadata": call_data.get("metadata", {})
            })

        elif event_type == "call_ended":
            logger.info(f"🔄 Processing ended call: {call_id}")
            # Update call status and trigger transcript processing
            await update_call_status(call_id, {
                "status": "ended",
                "end_timestamp": datetime.now().isoformat(),
                "disconnection_reason": call_data.get("disconnection_reason"),
                "metadata": call_data.get("metadata", {})
            })

            try:
                # Process the call asynchronously
                logger.info(f"📥 Fetching transcript for call: {call_id}")
                await fetch_and_store_retell_transcript(call_id)
                logger.info(f"✅ Successfully processed call: {call_id}")
            except Exception as e:
                logger.error(f"❌ Error processing call {call_id}: {str(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                await update_call_status(call_id, {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        elif event_type == "call_analyzed":
            logger.info(f"📊 Call analysis complete: {call_id}")
            # Update call status with analysis results
            await update_call_status(call_id, {
                "status": "analyzed",
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_data": call_data.get("call_analysis", {}),
                "metadata": call_data.get("metadata", {})
            })

        else:
            logger.warning(f"⚠️ Unknown event type: {event_type}")
            return {"status": "error", "message": f"Unknown event type: {event_type}"}, 400

        logger.info("="*50)
        return {"status": "success", "message": f"Successfully processed {event_type} event"}

    except Exception as e:
        logger.error(f"❌ Error in webhook handler: {str(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}, 500

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

async def fetch_retell_transcript(call_id: str) -> Optional[Dict]:
    """Fetch transcript from Retell API."""
    try:
        logger.info(f"🔄 Fetching transcript for call {call_id}")
        
        # Get call data from Retell
        call_data = await get_retell_call(call_id)
        if not call_data:
            raise Exception(f"Could not fetch call data for {call_id}")
            
        # Extract transcript
        transcript = call_data.get("transcript", "")
        if not transcript:
            raise Exception(f"No transcript available for call {call_id}")
            
        return {
            'transcript': transcript,
            'metadata': call_data.get('metadata', {}),
            'call_status': call_data.get('call_status'),
            'duration_ms': call_data.get('duration_ms')
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching transcript: {str(e)}")
        raise

async def fetch_and_store_retell_transcript(call_id: str) -> dict:
    """Fetch transcript from Retell and store in memory and Pinecone"""
    try:
        logger.info(f"🔄 Fetching transcript for call {call_id}")
        
        # Get transcript data
        transcript_data = await fetch_retell_transcript(call_id)
        if not transcript_data:
            raise Exception(f"Could not fetch transcript data for {call_id}")
            
        # Process with OpenAI
        processed_data = await process_transcript_with_openai(transcript_data['transcript'])
        
        # Store in memory
        call_statuses[call_id] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "metadata": transcript_data['metadata'],
            "processed_data": processed_data
        }
        
        # Store in Pinecone if we have a candidate ID
        candidate_id = transcript_data['metadata'].get('candidate_id')
        if candidate_id:
            logger.info(f"💾 Storing transcript in Pinecone for candidate {candidate_id}")
            await store_in_pinecone(candidate_id, processed_data)
        
        # Send email if we have the address
        email = transcript_data['metadata'].get('email')
        if email:
            logger.info(f"📧 Sending email to {email}")
            interaction_agent = InteractionAgent()
            email_result = interaction_agent.send_transcript_summary(email, processed_data)
            
            if email_result.get("status") == "success":
                logger.info(f"✅ Successfully sent email to {email}")
                call_statuses[call_id]["email_sent"] = True
                call_statuses[call_id]["email_sent_at"] = datetime.now().isoformat()
            else:
                logger.error(f"❌ Failed to send email: {email_result.get('error')}")
        
        logger.info(f"✅ Successfully processed call {call_id}")
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Error processing transcript for call {call_id}: {str(e)}")
        raise

async def get_retell_call(call_id: str) -> dict:
    """Fetch call data from Retell API"""
    try:
        logger.info(f"🔄 Fetching call data from Retell for {call_id}")
        async with aiohttp.ClientSession() as session:
            url = f"https://api.retellai.com/v2/get-call/{call_id}"
            headers = {
                "Authorization": f"Bearer {RETELL_API_KEY}"
            }
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch call data: {response.status}")
                return await response.json()
    except Exception as e:
        logger.error(f"❌ Error fetching call data: {str(e)}")
        raise

app = app