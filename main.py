from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from agents.brain_agent import BrainAgent
from agents.interaction_agent import InteractionAgent
from agents.vector_store import VectorStore
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime
import os
from dotenv import load_dotenv
import httpx
from enum import Enum
import openai
from pydantic import ValidationError
import json
import base64
from openai import AsyncOpenAI
import PyPDF2
import io
from pdf2image import convert_from_bytes
import tempfile
import requests
from typing import Annotated
import aiohttp
import traceback
import asyncio

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="Anita AI Recruitment API",
    description="API for AI-driven recruitment with enhanced candidate-job matching",
    version="2.0.0"
)

interaction_agent = InteractionAgent()
brain_agent = BrainAgent()
vector_store = VectorStore()

# Add Retell AI configuration
RETELL_API_KEY = os.getenv('RETELL_API_KEY')
RETELL_API_BASE = "https://api.retellai.com/v2"
RETELL_FROM_NUMBER = os.getenv('RETELL_FROM_NUMBER')
RETELL_AGENT_ID = os.getenv('RETELL_AGENT_ID')

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

class JobSubmission(BaseModel):
    id: str
    title: str
    company: str
    description: str
    role_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details including city, state, salary_range, work_authorization"
    )
    company_information: Dict[str, Any] = Field(
        default_factory=dict,
        description="Company details including culture, work environment"
    )

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

class CallStatusRequest(BaseModel):
    """Request model for checking call status"""
    candidate_id: str
    call_id: str

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

async def process_pdf_to_text(file: UploadFile) -> Dict[str, Any]:
    """Process a PDF file and extract its text and image content."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        # Read the uploaded file
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        # First try to extract text using PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            text = text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction failed: {str(e)}")
            text = ""

        # Convert PDF to images for OCR processing
        try:
            # Create a temporary directory for PDF processing
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_bytes(contents, output_folder=temp_dir)
                
                # Convert first page to base64 for vision processing
                if images:
                    img_byte_arr = io.BytesIO()
                    images[0].save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
                else:
                    base64_image = None
                    
                return {
                    "text": text,
                    "base64_image": base64_image,
                    "total_pages": len(images)
                }
        except Exception as e:
            print(f"PDF to image conversion failed: {str(e)}")
            return {
                "text": text,
                "base64_image": None,
                "total_pages": 0
            }
            
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
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare the content for GPT-4
        content = resume_data.get("text", "").strip()
        
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
            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Please analyze this resume and extract the key information:\n\n{content}"}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={ "type": "json_object" }
            )
            
            try:
                processed_data = json.loads(response.choices[0].message.content)
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

@app.post("/candidates", response_model=Dict[str, Any])
async def submit_candidate(
    name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    linkedin: Optional[str] = Form(None),
    resume: UploadFile = File(...)
):
    """
    Submit a new candidate.
    1. Process the PDF resume (both text and image)
    2. Analyze resume content with OpenAI
    3. Store candidate data in vector database
    """
    try:
        # Generate a unique candidate ID
        candidate_id = f"candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Format phone number to E.164 format
        phone = phone_number.strip()
        phone = '+' + ''.join(filter(str.isdigit, phone.replace('+', '')))
        if not phone.startswith('+'):
            phone = '+' + phone
        
        # Step 1: Process the PDF resume
        print(f"Processing resume: {resume.filename}")
        pdf_data = await process_pdf_to_text(resume)
        
        if not pdf_data.get("text") and not pdf_data.get("base64_image"):
            raise HTTPException(
                status_code=400,
                detail="Could not extract any content from the PDF. Please ensure the file is valid and contains readable content."
            )
        
        # Step 2: Process resume content through OpenAI
        print(f"Analyzing resume content for candidate {candidate_id}...")
        resume_analysis = await process_resume_text(pdf_data)
        
        if not resume_analysis.get("processed"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process resume content: {resume_analysis.get('error', 'Unknown error')}"
            )
        
        # Create comprehensive context from resume analysis
        context_data = {}
        if resume_analysis.get("structured_data"):
            data = resume_analysis["structured_data"]
            
            # Add skills
            if data.get("skills"):
                if isinstance(data["skills"], list):
                    context_data["skills"] = ", ".join(data["skills"][:10])
                else:
                    context_data["skills"] = str(data["skills"])
            
            # Add experience summary
            if data.get("experience"):
                if isinstance(data["experience"], list):
                    exp_summary = "; ".join([str(exp) for exp in data["experience"][:3]])
                    context_data["experience"] = exp_summary
                else:
                    context_data["experience"] = str(data["experience"])
            
            # Add education
            if data.get("education"):
                if isinstance(data["education"], list):
                    edu_summary = "; ".join([str(edu) for edu in data["education"]])
                    context_data["education"] = edu_summary
                else:
                    context_data["education"] = str(data["education"])
            
            # Add professional summary
            if data.get("summary"):
                context_data["professional_summary"] = str(data["summary"])
            
            # Add achievements
            if data.get("achievements"):
                if isinstance(data["achievements"], list):
                    achievements_summary = "; ".join([str(ach) for ach in data["achievements"][:3]])
                    context_data["achievements"] = achievements_summary
                else:
                    context_data["achievements"] = str(data["achievements"])
        
        # Create a comprehensive context string
        context_parts = []
        for key, value in context_data.items():
            if value:
                context_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # Store candidate data
        print(f"Storing candidate data for {candidate_id}...")
        candidate_data = {
            "name": name,
            "email": email,
            "phone_number": phone,
            "linkedin": linkedin,
            "resume_text": pdf_data.get("text", ""),
            "resume_analysis": resume_analysis,
            "context": " | ".join(context_parts)
        }
        
        try:
            vector_result = vector_store.store_candidate(candidate_id, candidate_data)
            if vector_result.get('status') != 'success':
                print(f"Warning: Failed to store candidate data: {vector_result.get('message')}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to store candidate data: {vector_result.get('message')}"
                )
        except Exception as e:
            print(f"Error storing candidate data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store candidate data: {str(e)}"
            )
        
        return {
            "status": "success",
            "message": "Candidate profile created successfully",
            "candidate_id": candidate_id,
            "profile": {
                "name": name,
                "email": email,
                "phone_number": phone,
                "linkedin": linkedin
            },
            "resume_analysis": resume_analysis.get("structured_data", {}),
            "stored_in_vector_db": True
        }
                
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in submit_candidate: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

async def process_transcript_with_openai(transcript: str) -> Dict[str, Any]:
    """Process transcript using OpenAI to extract structured information."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured"
        )

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert at analyzing interview transcripts and extracting key information.
                Please analyze the transcript and extract the following information in JSON format:
                {
                    "candidate_name": string,
                    "linkedin_url": string,
                    "contact_information": string,
                    "date_of_call": string (ISO format),
                    "current_role": string,
                    "current_company": string,
                    "previous_companies": string[],
                    "tech_stack": string[],
                    "years_of_experience": number,
                    "industries": string[],
                    "undesired_industries": string[],
                    "company_size_at_join": number,
                    "current_company_size": number,
                    "company_stage": string,
                    "experience_with_significant_company_growth": boolean,
                    "early_stage_startup_experience": boolean,
                    "leadership_experience": boolean,
                    "preferred_work_arrangement": string[],
                    "preferred_locations": [{"city": string, "state": string}],
                    "visa_sponsorship_needed": boolean,
                    "salary_expectations": {
                        "min": number,
                        "max": number
                    },
                    "desired_company_stage": string[],
                    "preferred_industries": string[],
                    "preferred_product_types": string[],
                    "motivation_for_job_change": string[],
                    "work_life_balance_preferences": string,
                    "desired_company_culture": string,
                    "traits_to_avoid_detected": string[],
                    "additional_notes": string,
                    "matched_role": string | null,
                    "match_score": number | null,
                    "match_decision": string | null,
                    "candidate_tags": string[],
                    "next_steps": string,
                    "role_preferences": string[],
                    "technologies_to_avoid": string[],
                    "company_culture_preferences": string[],
                    "work_environment_preferences": string[],
                    "career_goals": string[],
                    "skills_to_develop": string[],
                    "preferred_project_types": string[],
                    "company_mission_alignment": string[],
                    "preferred_company_size": string[],
                    "funding_stage_preferences": string[],
                    "total_compensation_expectations": {
                        "base_salary_min": number,
                        "base_salary_max": number,
                        "equity": string,
                        "bonus": string
                    },
                    "benefits_preferences": string[],
                    "deal_breakers": string[],
                    "bad_experiences_to_avoid": string[],
                    "willing_to_relocate": boolean,
                    "preferred_interview_process": string[],
                    "company_reputation_importance": string,
                    "preferred_management_style": string[],
                    "industries_to_explore": string[],
                    "project_visibility_preference": string[]
                }

                Extract as much information as possible from the transcript. For fields where information is not available, use empty strings for string fields, empty arrays for array fields, 0 for number fields, false for boolean fields, and null for nullable fields.
                
                For salary and compensation fields, convert any mentioned annual amounts to numbers. For equity and bonus fields in total_compensation_expectations, capture the details as strings.
                
                For location preferences, try to structure them as city/state pairs. If only a city or region is mentioned, include what is available.
                
                Pay special attention to:
                1. Tech stack and technical skills
                2. Career motivations and preferences
                3. Deal breakers and red flags
                4. Compensation expectations
                5. Work style and culture fit indicators"""},
                {"role": "user", "content": f"Please analyze this transcript and extract the information: {transcript}"}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        # Extract the JSON response
        processed_data = response.choices[0].message.content
        
        # If the response is a string containing JSON, parse it
        if isinstance(processed_data, str):
            import json
            try:
                processed_data = json.loads(processed_data)
            except json.JSONDecodeError:
                raise ValueError("Failed to parse OpenAI response as JSON")
        
        # Add the raw transcript to the processed data
        processed_data["raw_transcript"] = transcript
        
        return processed_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process transcript with OpenAI: {str(e)}"
        )

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

@app.post("/jobs/submit", response_model=Dict[str, Any])
async def submit_job(job: JobSubmission):
    """
    Submit a new job posting.
    
    The job will be vectorized and stored in the jobs index,
    ready for matching with candidates.
    """
    try:
        # Create text representation and store in jobs index
        vector = vector_store.get_embedding(
            f"{job.title} {job.description}"
        )
        
        vector_store.jobs_index.upsert(vectors=[(
            job.id,
            vector,
            {
                "title": job.title,
                "company": job.company,
                "description": job.description,
                "role_details": job.role_details,
                "company_information": job.company_information,
                "timestamp": datetime.utcnow().isoformat()
            }
        )])
        
        return {
            "status": "success",
            "message": f"Job {job.id} stored successfully",
            "job_id": job.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/jobs/open-positions", response_model=List[Dict[str, Any]])
async def list_open_positions():
    """
    List all available job positions.
    
    Returns a list of all jobs currently stored in the system,
    including their full details and requirements.
    """
    try:
        return brain_agent.matching_agent.fetch_open_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/candidates/match-jobs", response_model=MatchResponse)
async def match_jobs_to_candidate(candidate_id: str, top_k: Optional[int] = 5):
    """
    Find jobs that match a specific candidate.
    
    Enhanced matching considers:
    - Semantic similarity of skills and experience
    - Location preferences
    - Work environment preferences
    - Compensation requirements
    - Work authorization requirements
    
    Returns detailed match information including match scores and reasons.
    """
    result = vector_store.find_similar_jobs(candidate_id, top_k)
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['message'])
    
    return {
        "status": "success",
        "matches": result['matches'],
        "total_matches": len(result['matches'])
    }

@app.get("/candidates/{candidate_id}/profile", response_model=Dict[str, Any])
async def get_candidate_profile(candidate_id: str):
    """
    Retrieve a candidate's full profile including match history.
    
    Returns the candidate's information, preferences, and any matching results.
    """
    profile = brain_agent.candidate_profiles.get(candidate_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return {
        "status": "success",
        "profile": profile,
        "current_state": brain_agent.state.get(candidate_id, "unknown")
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

async def delete_retell_knowledge_base(knowledge_base_id: str) -> bool:
    """Delete a knowledge base from Retell AI."""
    if not RETELL_API_KEY:
        raise HTTPException(status_code=500, detail="Retell AI API key not configured")

    try:
        print(f"\n=== Deleting Knowledge Base {knowledge_base_id} ===")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"https://api.retellai.com/delete-knowledge-base/{knowledge_base_id}",
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                }
            )

            print(f"Delete response status: {response.status_code}")
            print(f"Delete response body: {response.text}")

            if response.status_code == 204:
                print(f"Successfully deleted knowledge base {knowledge_base_id}")
                return True
            elif response.status_code == 404:
                print(f"Knowledge base {knowledge_base_id} not found - considering it already deleted")
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
    Also cleans up by deleting the knowledge base after the call is completed.
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
            knowledge_base_id = None
            
            if retell_data.get('call_status') == RetellCallStatus.ENDED:
                print("\nCall has ended - attempting to clean up knowledge base")
                # Try to get knowledge_base_id from multiple possible locations
                metadata = retell_data.get('metadata', {})
                knowledge_base_id = (
                    metadata.get('knowledge_base_id') or
                    retell_data.get('knowledge_base_id') or
                    metadata.get('knowledgeBaseId')
                )
                
                if knowledge_base_id:
                    print(f"Found knowledge base ID: {knowledge_base_id}")
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            delete_response = await client.delete(
                                f"https://api.retellai.com/delete-knowledge-base/{knowledge_base_id}",
                                headers={
                                    "Authorization": f"Bearer {RETELL_API_KEY}",
                                    "Content-Type": "application/json"
                                }
                            )
                            
                            print(f"Delete response status: {delete_response.status_code}")
                            if delete_response.status_code in (204, 404):
                                knowledge_base_cleaned = True
                                print("Successfully cleaned up knowledge base")
                            else:
                                print(f"Failed to delete knowledge base: {delete_response.text}")
                    except Exception as e:
                        print(f"Error deleting knowledge base: {str(e)}")
                else:
                    print("No knowledge base ID found in response")
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
                    "knowledge_base_id": knowledge_base_id
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
                    "knowledge_base_id": knowledge_base_id
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

            if response.status_code != 201:
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

@app.post("/api/makeCall", response_model=Dict[str, Any])
async def make_call(
    name: Annotated[str, Form()],
    email: Annotated[str, Form()],
    phone_number: Annotated[str, Form()],
    linkedin: Annotated[str, Form()],
    resume_file: UploadFile = File(...)
):
    """Make a call using Retell AI with enhanced error handling and logging."""
    if not all([RETELL_API_KEY, RETELL_API_BASE, RETELL_FROM_NUMBER, RETELL_AGENT_ID]):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retell AI configuration incomplete"
        )

    try:
        print("\n=== Starting Make Call Process ===")
        print("API configuration validated successfully")

        # Generate a unique candidate ID
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        candidate_id = f"candidate_{timestamp}"
        print(f"Generated candidate ID: {candidate_id}")

        # Format phone number
        formatted_phone = format_phone_number(phone_number)
        print(f"Formatted phone number: {formatted_phone}")

        # Validate resume file
        if not resume_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        print(f"Validated resume file: {resume_file.filename}")

        # Create knowledge base
        print("Creating Retell AI knowledge base...")
        knowledge_base_id = await create_retell_knowledge_base(
            resume_file,
            name
        )
        print(f"Created knowledge base with ID: {knowledge_base_id}")

        # Prepare dynamic variables
        dynamic_vars = {
            "user_name": name,
            "email": email,
            "phone_number": formatted_phone,
            "linkedin": linkedin or "Not provided"
        }
        print(f"Dynamic variables prepared: {dynamic_vars}")

        print("Initiating call with Retell AI...")
        # Prepare the payload for Retell AI
        payload = {
            "to_number": formatted_phone,
            "from_number": RETELL_FROM_NUMBER,
            "override_agent_id": RETELL_AGENT_ID,
            "dynamic_variables": dynamic_vars,
            "knowledge_base_id": knowledge_base_id,
            "webhook_url": "https://anita-fastapi.onrender.com/webhook/retell",
            "metadata": {
                "candidate_id": candidate_id,
                "name": name,
                "email": email,
                "linkedin": linkedin or "Not provided",
                "source": "anita_ai",
                "knowledge_base_id": knowledge_base_id
            }
        }
        print(f"Prepared Retell AI payload: {json.dumps(payload, indent=2)}")

        # Make the API call to Retell AI
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{RETELL_API_BASE}/calls",
                json=payload,
                headers={
                    "Authorization": f"Bearer {RETELL_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            print(f"Retell AI response status: {response.status_code}")
            print(f"Retell AI response headers: {response.headers}")
            print(f"Retell AI response body: {response.text}")

            if response.status_code != 201:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Retell AI service error: {response.text}"
                )

            response_data = response.json()
            return {
                "status": "success",
                "call_id": response_data["call_id"],
                "candidate_id": candidate_id,
                "knowledge_base_id": knowledge_base_id
            }

    except HTTPException as he:
        print(f"HTTPException in make_call: {he.status_code}: {he.detail}")
        raise
    except Exception as e:
        print(f"Unexpected error in make_call: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        print("=== Make Call Process Complete ===\n")

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

@app.post("/webhook/retell")
async def retell_webhook(request: Request):
    """Handle Retell AI webhooks for call status updates."""
    try:
        webhook_data = await request.json()
        print(f"\n=== Processing Retell Webhook ===")
        print(f"Webhook data: {json.dumps(webhook_data, indent=2)}")
        
        call_status = webhook_data.get('call_status')
        if call_status == 'ended':
            # Extract knowledge base ID from metadata
            metadata = webhook_data.get('metadata', {})
            knowledge_base_id = metadata.get('knowledge_base_id')
            
            if knowledge_base_id:
                print(f"Call ended - cleaning up knowledge base {knowledge_base_id}")
                success = await delete_retell_knowledge_base(knowledge_base_id)
                if success:
                    print("Successfully cleaned up knowledge base via webhook")
                else:
                    print("Failed to clean up knowledge base via webhook")
            else:
                print("No knowledge base ID found in webhook metadata")
        
        return {"status": "success", "message": "Webhook processed"}
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

