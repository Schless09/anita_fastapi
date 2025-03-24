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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
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

# Initialize in-memory storage for call statuses
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
print("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
job_index = pc.Index("job-details")

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
    timestamp: datetime
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

        Format the response as a valid JSON object with these exact keys."""

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
                        "summary": "Failed to parse structured data from resume"
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
    """Process a call transcript with OpenAI to extract key information."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured"
        )

    try:
        # Format transcript for analysis
        formatted_transcript = f"""Please analyze this interview transcript and extract key information:

{transcript}

Please provide the analysis in JSON format with the following structure:
{{
    "candidate_name": string,
    "linkedin_url": string | null,
    "contact_information": string,
    "current_role": string,
    "years_of_experience": number,
    "tech_stack": string[],
    "key_skills": string[],
    "experience_highlights": string[],
    "education": {{
        "degree": string,
        "institution": string,
        "year": string
    }},
    "preferred_work_arrangement": string,
    "preferred_locations": [{{
        "city": string,
        "state": string
    }}],
    "salary_expectations": {{
        "min": number,
        "max": number
    }},
    "availability": string,
    "interests": string[],
    "next_steps": string,
    "call_summary": string
}}"""

        # Get analysis from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing interview transcripts and extracting structured information."},
                {"role": "user", "content": formatted_transcript}
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={ "type": "json_object" }
        )

        # Parse the response
        analysis = json.loads(response.choices[0].message.content)
        
        # Add the raw transcript
        analysis["raw_transcript"] = transcript
        
        return analysis

    except Exception as e:
        logger.error(f"Error processing transcript with OpenAI: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "raw_transcript": transcript
        }

def format_transcript_for_openai(transcript: str) -> str:
    """Format a transcript for OpenAI analysis."""
    # Remove any special characters or formatting
    cleaned = re.sub(r'[\r\n]+', '\n', transcript)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Truncate if too long (context window limitation)
    max_length = 14000  # Leave room for system message and response
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned

async def sync_call_statuses():
    """Synchronize call statuses with Retell API."""
    try:
        logger.info("Starting call status synchronization")
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get all calls from the last 24 hours
            response = await client.get(
                f"{RETELL_API_BASE}/list-calls",
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                },
                params={
                    "start_time": int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000),
                    "end_time": int(datetime.utcnow().timestamp() * 1000)
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch calls: {response.text}")
                return
            
            calls_data = response.json()
            for call in calls_data.get("calls", []):
                call_id = call.get("call_id")
                if call_id:
                    # Update our local status
                    call_statuses[call_id] = {
                        "status": call.get("call_status", "unknown"),
                        "timestamp": call.get("start_timestamp"),
                        "metadata": call.get("metadata", {})
                    }
            
            logger.info(f"Synchronized {len(calls_data.get('calls', []))} call statuses")
            
    except Exception as e:
        logger.error(f"Error syncing call statuses: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")

def clean_test_calls() -> int:
    """Remove test calls from memory. Returns number of calls removed."""
    try:
        initial_count = len(call_statuses)
        # Remove calls older than 24 hours
        cutoff_time = datetime.utcnow() - timedelta(days=1)
        
        to_remove = []
        for call_id, data in call_statuses.items():
            try:
                call_time = datetime.fromisoformat(data.get("timestamp", "").replace("Z", "+00:00"))
                if call_time < cutoff_time:
                    to_remove.append(call_id)
            except (ValueError, AttributeError):
                # If we can't parse the timestamp, assume it's old
                to_remove.append(call_id)
        
        # Remove the identified calls
        for call_id in to_remove:
            del call_statuses[call_id]
        
        removed_count = initial_count - len(call_statuses)
        logger.info(f"Removed {removed_count} old calls")
        return removed_count
        
    except Exception as e:
        logger.error(f"Error cleaning test calls: {str(e)}")
        return 0

async def update_call_status(call_id: str, status_data: Dict[str, Any]):
    """Update call status in memory and Pinecone."""
    try:
        # Update in memory
        call_statuses[call_id] = status_data
        
        # Update in Pinecone
        vector_store = VectorStore()
        await vector_store.update_call_status(call_id, status_data)
        
        logger.info(f"Updated status for call {call_id}")
        
    except Exception as e:
        logger.error(f"Error updating call status: {str(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")

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

@app.get("/candidates/{candidate_id}/profile")
async def get_candidate_profile(candidate_id: str):
    """Get a candidate's complete profile including processed resume data."""
    try:
        vector_store = VectorStore(init_openai=True)
        profile = vector_store.get_candidate_profile(candidate_id)
        
        if profile["status"] == "error":
            raise HTTPException(
                status_code=404,
                detail=profile["message"]
            )
            
        return profile
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

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

async def delete_retell_knowledge_base(knowledge_base_id: str) -> bool:
    """Delete a knowledge base from Retell AI."""
    if not RETELL_API_KEY:
        raise HTTPException(status_code=500, detail="Retell AI API key not configured")

    try:
        print(f"\n=== Deleting Knowledge Base {knowledge_base_id} ===")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{RETELL_API_BASE}/knowledge-bases/{knowledge_base_id}",
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                }
            )

            print(f"Delete response status: {response.status_code}")
            print(f"Delete response body: {response.text}")

            if response.status_code == 404:
                print(f"Knowledge base {knowledge_base_id} not found - considering it already deleted")
                return True
            elif response.status_code == 200:
                print(f"Successfully deleted knowledge base {knowledge_base_id}")
                return True
            else:
                print(f"Failed to delete knowledge base: {response.text}")
                return False

    except Exception as e:
        print(f"Error deleting knowledge base: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return False
    finally:
        print("=== Knowledge Base Deletion Complete ===\n")

@app.post("/candidate/retell-transcript", response_model=Dict[str, Any])
async def fetch_and_store_retell_transcript(call_data: RetellCallData):
    """
    Fetch transcript from Retell AI call and store it in the candidate's profile.
    Also cleans up by deleting the knowledge base source after the call is completed.
    """
    if not RETELL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Configuration error",
                "message": "Retell AI API key not configured",
                "action_required": "Please set RETELL_API_KEY in environment variables"
            }
        )
    
    try:
        print(f"\n=== Processing Transcript for Call {call_data.call_id} ===")
        # Fetch call transcript from Retell AI
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("Fetching transcript from Retell AI...")
            response = await client.get(
                f"https://api.retellai.com/get-call/{call_data.call_id}",
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "error": "Authentication failed",
                        "message": "Invalid Retell AI API key",
                        "action_required": "Please check your API key configuration"
                    }
                )
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Call not found",
                        "message": f"No call found with ID: {call_data.call_id}",
                        "action_required": "Please verify the call ID"
                    }
                )
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "Retell AI service error",
                        "message": f"Failed to fetch transcript: {response.text}",
                        "action_required": "Please try again later"
                    }
                )
            
            print("Successfully fetched transcript")
            retell_data = response.json()
            
            # Process the call data
            processed_data = await process_transcript_with_openai(retell_data['transcript'])
            print("Processed transcript with OpenAI")
            
            # Handle knowledge base cleanup for completed calls
            knowledge_base_cleaned = False
            knowledge_base_id = "knowledge_base_b1df2fc51182f47b"  # Fixed knowledge base ID
            source_id = retell_data.get('metadata', {}).get('source_id')  # Get source_id from metadata

            # Check if call has ended and attempt knowledge base cleanup
            if retell_data.get('call_status') == RetellCallStatus.ENDED and source_id:
                print("\nCall has ended - attempting to clean up knowledge base source")
                print(f"Found knowledge base ID: {knowledge_base_id}")
                print(f"Found source ID: {source_id}")
                try:
                    # Delete the specific source from the knowledge base
                    delete_response = await client.delete(
                        f"https://api.retellai.com/delete-knowledge-base-source/{knowledge_base_id}/source/{source_id}",
                        headers={
                            "Authorization": f"Bearer {RETELL_API_KEY}",
                            "Content-Type": "application/json"
                        }
                    )
                    
                    print(f"Delete response status: {delete_response.status_code}")
                    if delete_response.status_code in (200, 404):
                        knowledge_base_cleaned = True
                        print("Successfully cleaned up knowledge base source")
                except Exception as e:
                    print(f"Error deleting knowledge base source: {str(e)}")
                    print(f"Error type: {type(e)}")
                    print(f"Error traceback: {traceback.format_exc()}")
            else:
                print(f"Call status is {retell_data.get('call_status')} - skipping knowledge base cleanup")
            
            # Create enhanced transcript
            enhanced_transcript = EnhancedTranscript(
                raw_transcript=retell_data['transcript'],
                call_summary=retell_data.get('call_analysis', {}).get('call_summary', ''),
                user_sentiment=retell_data.get('call_analysis', {}).get('user_sentiment', 'Unknown'),
                call_successful=retell_data.get('call_analysis', {}).get('call_successful', False),
                custom_analysis=retell_data.get('call_analysis', {}).get('custom_analysis_data', {}),
                timestamp=datetime.utcnow(),
                call_status=retell_data['call_status'],
                call_duration=calculate_call_duration(retell_data),
                error_details=retell_data.get('disconnection_reason')
            )
            
            # Update candidate profile
            print("Updating candidate profile...")
            result = brain_agent.add_transcript_to_profile(
                call_data.candidate_id,
                {
                    "raw_transcript": retell_data['transcript'],
                    "processed_data": processed_data,
                    "enhanced_transcript": enhanced_transcript.dict(),
                    "knowledge_base_cleaned": knowledge_base_cleaned,
                    "knowledge_base_id": knowledge_base_id,
                    "source_id": source_id
                }
            )
            
            return {
                "status": "success",
                "message": "Transcript processed and stored successfully",
                "transcript_data": enhanced_transcript.dict(),
                "candidate_state": result.get('current_state'),
                "call_id": call_data.call_id,
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "call_duration_seconds": enhanced_transcript.call_duration,
                    "sentiment": enhanced_transcript.user_sentiment,
                    "success_status": enhanced_transcript.call_successful,
                    "knowledge_base_cleaned": knowledge_base_cleaned,
                    "knowledge_base_id": knowledge_base_id,
                    "source_id": source_id
                }
            }
            
    except Exception as e:
        print(f"Error in fetch_and_store_retell_transcript: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "action_required": "Please contact support if the issue persists"
            }
        )
    finally:
        print("=== Transcript Processing Complete ===\n")

async def cleanup_completed_calls():
    """Background task to check for completed calls and process them."""
    while True:
        try:
            print("\n=== Starting cleanup of completed calls ===")
            
            # Get all calls that have ended but haven't been processed
            ended_calls = {
                call_id: data for call_id, data in call_statuses.items()
                if data["status"] == RetellCallStatus.ENDED and not data.get("processed", False)
            }
            
            if not ended_calls:
                print("No completed calls to process")
            else:
                print(f"Found {len(ended_calls)} completed calls to process")
                
                for call_id, call_data in ended_calls.items():
                    print(f"\nProcessing call {call_id}")
                    
                    try:
                        # Fetch transcript
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(
                                f"{RETELL_API_BASE}/get-call/{call_id}",
                                headers={
                                    "Authorization": f"Bearer {RETELL_API_KEY}",
                                    "Content-Type": "application/json"
                                }
                            )
                            
                            if response.status_code == 404:
                                print("No transcript available - call may have ended prematurely")
                                call_data["transcript_status"] = "No transcript available - call may have ended prematurely"
                            elif response.status_code == 200:
                                retell_data = response.json()
                                if "transcript" in retell_data:
                                    # Process transcript
                                    processed_data = await process_transcript_with_openai(retell_data["transcript"])
                                    
                                    # Send email if we have a candidate email and haven't sent one yet
                                    if call_data.get("candidate_email") and not call_data.get("email_sent", False):
                                        email_result = interaction_agent.send_transcript_summary(
                                            call_data["candidate_email"],
                                            processed_data
                                        )
                                        # Update email status
                                        call_data["email_sent"] = True
                                        call_data["email_sent_at"] = datetime.utcnow().isoformat()
                                        call_data["email_status"] = email_result
                                    
                                    call_data["transcript_processed"] = True
                                else:
                                    print("No transcript in response")
                                    call_data["transcript_status"] = "No transcript available in response"
                            else:
                                print(f"Error fetching transcript: {response.status_code}")
                                call_data["transcript_status"] = f"Error fetching transcript: {response.status_code}"
                        
                        # Mark call as processed
                        call_data["processed"] = True
                        call_data["processed_at"] = datetime.utcnow().isoformat()
                        
                        # Update call status in memory and Pinecone
                        await update_call_status(call_id, call_data)
                        
                    except Exception as e:
                        print(f"Error processing call {call_id}: {str(e)}")
                        print(f"Error type: {type(e)}")
                        print(f"Error traceback: {traceback.format_exc()}")
            
            print("=== Completed cleanup of completed calls ===\n")
            
            # Wait 30 seconds before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            # Wait a minute before retrying if there's an error
            await asyncio.sleep(60)

def calculate_call_duration(retell_data: Dict[str, Any]) -> Optional[float]:
    """Calculate call duration from timestamps."""
    start_time = retell_data.get('start_timestamp')
    end_time = retell_data.get('end_timestamp')
    if start_time and end_time:
        return (end_time - start_time) / 1000  # Convert to seconds
    return None

async def create_retell_knowledge_base(resume_file: UploadFile, name: str) -> str:
    """Create a knowledge base in Retell AI with the resume file."""
    if not RETELL_API_KEY:
        raise HTTPException(status_code=500, detail="Retell AI API key not configured")

    try:
        print("\n=== Starting Knowledge Base Creation ===")
        print(f"Processing file: {resume_file.filename}")
        
        # Read the file content
        file_content = await resume_file.read()
        
        # Create a temporary file to send to Retell AI
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            print(f"Created temporary file: {temp_file.name}")
            temp_file.write(file_content)
            temp_file.flush()
            
            kb_name = f"KB_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            print(f"Knowledge base name: {kb_name}")

            # Make the request to create knowledge base
            async with httpx.AsyncClient() as client:
                print("\nSending request to Retell AI...")
                
                # Create form data
                files = {
                    'knowledge_base_files': (
                        resume_file.filename,
                        open(temp_file.name, 'rb'),
                        'application/pdf'
                    )
                }
                
                response = await client.post(
                    "https://api.retellai.com/create-knowledge-base",
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}"
                    },
                    data={
                        'knowledge_base_name': kb_name
                    },
                    files=files
                )

                print(f"\nResponse status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")

            # Clean up the temporary file
            os.unlink(temp_file.name)
            print(f"\nCleaned up temporary file: {temp_file.name}")

            if response.status_code not in (200, 201):
                print(f"Error creating knowledge base: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to create Retell AI knowledge base: {response.text}"
                )

            try:
                kb_data = response.json()
                kb_id = kb_data.get('knowledge_base_id')
                if not kb_id:
                    raise ValueError("No knowledge_base_id in response")
                print(f"\nSuccessfully created knowledge base with ID: {kb_id}")
                print("=== Knowledge Base Creation Complete ===\n")
                return kb_id
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse Retell AI response: {str(e)}"
                )

    except Exception as e:
        print(f"\nERROR in create_retell_knowledge_base: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge base: {str(e)}")

async def add_to_knowledge_base(resume: UploadFile, knowledge_base_id: str) -> Tuple[bool, Optional[str]]:
    """Add a resume to the Retell knowledge base and return success status and source ID."""
    try:
        print(f"\n=== Adding file to knowledge base {knowledge_base_id} ===")
        print(f"File name: {resume.filename}")
        
        # Read the file content
        file_content = await resume.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            print(f"Created temporary file: {temp_file.name}")
            
            try:
                # Initialize Retell client
                retell_client = Retell(api_key=RETELL_API_KEY)
                
                # Open the file and add it to the knowledge base
                with open(temp_file.name, "rb") as file:
                    print("Adding file to knowledge base using Retell client...")
                    response = retell_client.knowledge_base.add_sources(
                        knowledge_base_id=knowledge_base_id,
                        knowledge_base_files=[file]
                    )
                    
                    print(f"Response from Retell: {response}")
                    
                    # Extract source_id from the first source in knowledge_base_sources
                    if (hasattr(response, 'knowledge_base_sources') and 
                        response.knowledge_base_sources and 
                        hasattr(response.knowledge_base_sources[0], 'source_id')):
                        source_id = response.knowledge_base_sources[0].source_id
                        print(f"Successfully added file with source_id: {source_id}")
                        return True, source_id
                    else:
                        print("No source_id found in response structure")
                        print(f"Response structure: {dir(response)}")
                        if hasattr(response, 'knowledge_base_sources'):
                            print(f"Sources: {response.knowledge_base_sources}")
                        return False, None
            
            except Exception as e:
                print(f"Error adding file to knowledge base: {str(e)}")
                print(f"Error type: {type(e)}")
                print(f"Error traceback: {traceback.format_exc()}")
                return False, None
            
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                    print(f"Cleaned up temporary file: {temp_file.name}")
                except Exception as e:
                    print(f"Error cleaning up temporary file: {str(e)}")
    except Exception as e:
        print(f"Error in add_to_knowledge_base: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return False, None

@app.post("/api/makeCall")
async def make_call(
    candidate_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    resume: UploadFile = File(...)
):
    """Create a new Retell AI call for a candidate"""
    try:
        # Validate API configuration
        if not RETELL_API_KEY or not RETELL_AGENT_ID or not RETELL_FROM_NUMBER:
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
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phone number format: {str(e)}"
            )

        # Validate resume file
        if not resume.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Resume must be a PDF file"
            )

        # Use the fixed knowledge base ID
        knowledge_base_id = "knowledge_base_b1df2fc51182f47b"

        # Add the resume to the knowledge base and get the source ID
        success, source_id = await add_to_knowledge_base(resume, knowledge_base_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add resume to knowledge base"
            )

        # Prepare the request object
        retell_payload = {
            "from_number": RETELL_FROM_NUMBER,
            "to_number": formatted_number,
            "agent_id": RETELL_AGENT_ID,
            "knowledge_base_id": knowledge_base_id,
            "metadata": {
                "candidate_id": candidate_id,
                "name": name,
                "email": email,
                "source_id": source_id  # Add source_id to metadata
            },
            "retell_llm_dynamic_variables": {
                "candidate_name": name,
                "candidate_email": email
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
                print(f"Error creating call: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to create call: {error_detail}"
                )

            call_data = response.json()
            call_id = call_data.get("call_id")
            
            if not call_id:
                raise HTTPException(
                    status_code=500,
                    detail="No call_id in response"
                )

            # Register call status with source_id
            call_statuses[call_id] = {
                "status": "registered",
                "candidate_id": candidate_id,
                "source_id": source_id,  # Store source_id in call status
                "timestamp": datetime.utcnow().isoformat()
            }

            return {
                "message": "Call initiated successfully",
                "call_id": call_id,
                "status": "registered",
                "source_id": source_id  # Include source_id in response
            }

    except Exception as e:
        print(f"Error in make_call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create call: {str(e)}"
        )

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

@app.post("/webhook/retell")
async def retell_webhook(payload: RetellWebhookPayload):
    """Handle webhooks from Retell AI."""
    try:
        print(f"\n=== Processing Retell Webhook for Call {payload.call_id} ===")
        print(f"Call Status: {payload.call_status}")
        print(f"Webhook Metadata: {payload.metadata}")
        
        # Verify this is a call we know about
        if payload.call_id not in call_statuses:
            print(f"Warning: Received webhook for unknown call {payload.call_id}")
            return {"status": "success", "message": "Webhook processed"}
        
        # Get our stored call data
        call_data = call_statuses[payload.call_id]
        print(f"Stored call data: {call_data}")
        
        # Update call status
        call_data["status"] = payload.call_status
        
        # If call has ended, process transcript and send email
        if payload.call_status == RetellCallStatus.ENDED:
            try:
                # Fetch transcript
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{RETELL_API_BASE}/get-call/{payload.call_id}",
                        headers={
                            "Authorization": f"Bearer {RETELL_API_KEY}",
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code == 404:
                        print("No transcript available - call may have ended prematurely")
                        call_data["transcript_status"] = "No transcript available - call may have ended prematurely"
                    elif response.status_code == 200:
                        retell_data = response.json()
                        if "transcript" in retell_data:
                            # Process transcript
                            processed_data = await process_transcript_with_openai(retell_data["transcript"])
                            
                            # Send email if we have a candidate email
                            if call_data.get("candidate_email"):
                                email_result = interaction_agent.send_transcript_summary(
                                    call_data["candidate_email"],
                                    processed_data
                                )
                                # Update email status
                                call_data["email_sent"] = True
                                call_data["email_sent_at"] = datetime.utcnow().isoformat()
                                call_data["email_status"] = email_result
                            
                            call_data["transcript_processed"] = True
                        else:
                            print("No transcript in response")
                            call_data["transcript_status"] = "No transcript available in response"
                    else:
                        print(f"Error fetching transcript: {response.status_code}")
                        call_data["transcript_status"] = f"Error fetching transcript: {response.status_code}"
                
            except Exception as e:
                print(f"Error processing transcript: {str(e)}")
                call_data["transcript_status"] = f"Error processing transcript: {str(e)}"
        
        # Update call status in memory and Pinecone
        await update_call_status(payload.call_id, call_data)
        
        return {
            "status": "success",
            "message": "Webhook processed successfully",
            "call_id": payload.call_id,
            "call_status": payload.call_status,
            "transcript_status": call_data.get("transcript_status"),
            "email_status": call_data.get("email_status")
        }
        
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": str(e),
            "call_id": payload.call_id if hasattr(payload, 'call_id') else None
        }
    finally:
        print("=== Webhook Processing Complete ===\n")

async def delete_knowledge_base_source(knowledge_base_id: str, source_id: str) -> Tuple[bool, str]:
    """Delete a source from the knowledge base and verify the correct source was deleted."""
    try:
        # Initialize Retell client
        retell_client = Retell(api_key=RETELL_API_KEY)
        
        # First, get current state of knowledge base
        print(f"Checking current knowledge base state...")
        try:
            knowledge_bases = retell_client.knowledge_base.list()
            target_kb = next((kb for kb in knowledge_bases if kb.knowledge_base_id == knowledge_base_id), None)
            
            if not target_kb:
                return False, f"Knowledge base {knowledge_base_id} not found"
                
            initial_sources = target_kb.knowledge_base_sources
            print("\nCurrent sources in knowledge base:")
            for s in initial_sources:
                print(f"  - {s.source_id}: {s.filename}")
            
            source_exists = any(s.source_id == source_id for s in initial_sources)
            
            if not source_exists:
                print(f"Source {source_id} not found in knowledge base.")
                return True, "Source already deleted"
            
            # Attempt to delete the source using raw HTTP request for more control
            print(f"\nAttempting to delete source {source_id}...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                delete_url = f"https://api.retellai.com/v2/delete-knowledge-base-source/{knowledge_base_id}/source/{source_id}"
                delete_response = await client.delete(
                    delete_url,
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                
                if delete_response.status_code != 200:
                    print(f"Delete request failed with status {delete_response.status_code}")
                    return False, f"Delete request failed: {delete_response.text}"
                
                # Verify deletion by checking final state
                knowledge_bases = retell_client.knowledge_base.list()
                target_kb = next((kb for kb in knowledge_bases if kb.knowledge_base_id == knowledge_base_id), None)
                
                if not target_kb:
                    return False, "Failed to verify deletion - knowledge base not found"
                    
                final_sources = target_kb.knowledge_base_sources
                print("\nRemaining sources after deletion:")
                for s in final_sources:
                    print(f"  - {s.source_id}: {s.filename}")
                
                source_deleted = not any(s.source_id == source_id for s in final_sources)
                
                if source_deleted:
                    print(f"\nSuccessfully verified deletion of source {source_id}")
                    return True, "Success"
                else:
                    print("\nWarning: Source still exists after deletion attempt")
                    print("Initial sources:")
                    for s in initial_sources:
                        print(f"  - {s.source_id}: {s.filename}")
                    print("Final sources:")
                    for s in final_sources:
                        print(f"  - {s.source_id}: {s.filename}")
                        
                    return False, "Source still exists after deletion attempt"
                    
        except Exception as api_error:
            print(f"API Error: {str(api_error)}")
            return False, f"API Error: {str(api_error)}"
                
    except Exception as e:
        error_msg = f"Error deleting source {source_id}: {str(e)}"
        print(error_msg)
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return False, error_msg

async def check_call_status(call_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Check the status of a call with Retell API."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{RETELL_API_BASE}/get-call/{call_id}"
            print(f"Checking call status at: {url}")
            
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            print(f"Status check response code: {response.status_code}")
            if response.status_code == 200:
                call_info = response.json()
                status = call_info.get('call_status', 'unknown')
                print(f"Retrieved call status: {status}")
                return True, status, call_info
            elif response.status_code == 404:
                print(f"Call {call_id} not found")
                return False, "not_found", {}
            else:
                print(f"Error response: {response.text}")
                return False, f"error_{response.status_code}", {}
                
    except Exception as e:
        print(f"Error checking call status for {call_id}: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return False, "error", {}

class JobURL(BaseModel):
    url: str

async def scrape_job(url_data: JobURL) -> Dict[str, Any]:
    """
    Scrape job posting data from a provided URL using regular expressions.
    """
    try:
        # Validate URL
        parsed_url = urlparse(url_data.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL provided")

        # Fetch the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url_data.url, headers=headers)
        response.raise_for_status()
        text = response.text

        # Initialize job data dictionary
        job_data = {
            "company_name": None,
            "company_url": None,
            "job_title": None,
            "job_url": url_data.url,
            "salary_range": None,
            "work_arrangement": None,
            "city": [],
            "state": [],
            "tech_stack_must_haves": [],
            "tech_stack_nice_to_haves": [],
            "tech_stack_tags": [],
            "minimum_years_of_experience": None,
            "domain_expertise": [],
            "infrastructure_experience": [],
            "key_responsibilities": [],
            "company_culture": {
                "work_environment": "Not specified",
                "decision_making": "Not specified",
                "collaboration_style": "Not specified",
                "risk_tolerance": "Not specified",
                "values": "Not specified"
            }
        }

        # Extract company name from meta tags or title
        company_name_match = re.search(r'<meta[^>]*property="og:site_name"[^>]*content="([^"]*)"', text)
        if company_name_match:
            job_data['company_name'] = company_name_match.group(1)
        else:
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', text)
            if title_match:
                job_data['company_name'] = title_match.group(1).split(' - ')[0].strip()

        # Extract job title from h1 tag
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', text)
        if title_match:
            job_data['job_title'] = title_match.group(1).strip()

        # Salary range (look for common patterns)
        salary_patterns = [
            r'\$[\d,]+(?:\.\d{2})?\s*-\s*\$[\d,]+(?:\.\d{2})?',
            r'\$[\d,]+(?:\.\d{2})?\s*to\s*\$[\d,]+(?:\.\d{2})?',
            r'\$[\d,]+(?:\.\d{2})?\s*per\s*year'
        ]
        for pattern in salary_patterns:
            salary_match = re.search(pattern, text)
            if salary_match:
                job_data['salary_range'] = salary_match.group(0)
                break

        # Location
        location_match = re.search(r'location[:\s]+([^<]+)', text, re.I)
        if location_match:
            location_text = location_match.group(1).strip()
            # Split location into city and state
            parts = location_text.split(',')
            if len(parts) >= 2:
                job_data['city'] = [parts[0].strip()]
                job_data['state'] = [parts[1].strip()]

        # Work arrangement
        work_arrangement_keywords = {
            'remote': 'Remote',
            'hybrid': 'Hybrid',
            'on-site': 'On-site',
            'onsite': 'On-site'
        }
        for keyword, arrangement in work_arrangement_keywords.items():
            if re.search(keyword, text, re.I):
                job_data['work_arrangement'] = arrangement
                break

        # Technical requirements
        tech_keywords = {
            'must_have': ['python', 'javascript', 'java', 'react', 'node.js', 'aws', 'docker', 'kubernetes'],
            'nice_to_have': ['typescript', 'graphql', 'elasticsearch', 'terraform']
        }
        
        # Look for technical requirements section
        requirements_section = re.search(r'(?:requirements|qualifications|skills|tech stack)[:\s]+([^<]+)', text, re.I)
        if requirements_section:
            tech_text = requirements_section.group(1).lower()
            for category, keywords in tech_keywords.items():
                for keyword in keywords:
                    if keyword in tech_text:
                        if category == 'must_have':
                            job_data['tech_stack_must_haves'].append(keyword)
                        else:
                            job_data['tech_stack_nice_to_haves'].append(keyword)

        # Years of experience
        exp_match = re.search(r'(\d+)\+?\s*years?\s*of\s*experience', text, re.I)
        if exp_match:
            job_data['minimum_years_of_experience'] = exp_match.group(1)

        # Key responsibilities
        responsibilities_section = re.search(r'(?:responsibilities|duties|role)[:\s]+([^<]+)', text, re.I)
        if responsibilities_section:
            responsibilities_text = responsibilities_section.group(1)
            # Split into bullet points if they exist
            bullet_points = re.findall(r'[•\-\*]\s*([^\n]+)', responsibilities_text)
            if bullet_points:
                job_data['key_responsibilities'] = [point.strip() for point in bullet_points]
            else:
                # If no bullet points, split by newlines
                job_data['key_responsibilities'] = [line.strip() for line in responsibilities_text.split('\n') if line.strip()]

        return job_data

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing job data: {str(e)}")

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

app = app