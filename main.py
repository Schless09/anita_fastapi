from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from agents.brain_agent import BrainAgent
from agents.interaction_agent import InteractionAgent
from agents.vector_store import VectorStore
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from datetime import datetime
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

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize call status tracking
call_statuses = {}

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
    # Company Information
    company_name: Optional[str] = None
    company_url: Optional[str] = None
    company_stage: Optional[str] = None
    most_recent_funding_round_amount: Optional[str] = None
    total_funding_amount: Optional[str] = None
    investors: Optional[List[str]] = None
    team_size: Optional[str] = None
    founding_year: Optional[str] = None
    company_mission: Optional[str] = None
    target_market: Optional[List[str]] = None
    industry_vertical: Optional[str] = None
    company_vision: Optional[str] = None
    company_growth_story: Optional[str] = None
    company_culture: Optional[CompanyCulture] = None
    scaling_plans: Optional[str] = None
    mission_and_impact: Optional[str] = None
    tech_innovation: Optional[str] = None

    # Role Details
    job_title: Optional[str] = None
    job_url: Optional[str] = None
    positions_available: Optional[str] = None
    hiring_urgency: Optional[str] = None
    seniority_level: Optional[str] = None
    work_arrangement: Optional[str] = None
    city: Optional[List[str]] = None
    state: Optional[List[str]] = None
    visa_sponsorship: Optional[str] = None
    work_authorization: Optional[List[str]] = None
    salary_range: Optional[str] = None
    equity_range: Optional[str] = None
    reporting_structure: Optional[str] = None
    team_composition: Optional[str] = None
    role_status: Optional[str] = None

    # Technical Requirements
    role_category: Optional[str] = None
    tech_stack_must_haves: Optional[List[str]] = None
    tech_stack_nice_to_haves: Optional[List[str]] = None
    tech_stack_tags: Optional[List[str]] = None
    tech_breadth_requirement: Optional[str] = None
    minimum_years_of_experience: Optional[str] = None
    domain_expertise: Optional[List[str]] = None
    ai_ml_experience: Optional[str] = None
    infrastructure_experience: Optional[List[str]] = None
    system_design_level: Optional[str] = None
    coding_proficiency_required: Optional[str] = None
    coding_languages_versions: Optional[List[str]] = None
    version_control_experience: Optional[List[str]] = None
    ci_cd_tools: Optional[List[str]] = None
    collaborative_tools: Optional[List[str]] = None

    # Qualification Requirements
    leadership_requirement: Optional[str] = None
    education_requirement: Optional[str] = None
    advanced_degree_preference: Optional[str] = None
    papers_publications_preferred: Optional[str] = None
    prior_startup_experience: Optional[str] = None
    advancement_history_required: Optional[bool] = None
    independent_work_capacity: Optional[str] = None
    skills_must_have: Optional[List[str]] = None
    skills_preferred: Optional[List[str]] = None

    # Product and Role Context
    product_details: Optional[str] = None
    product_development_stage: Optional[str] = None
    technical_challenges: Optional[List[str]] = None
    key_responsibilities: Optional[List[str]] = None
    scope_of_impact: Optional[List[str]] = None
    expected_deliverables: Optional[List[str]] = None
    product_development_methodology: Optional[List[str]] = None

    # Startup Specific Factors
    stage_of_codebase: Optional[str] = None
    growth_trajectory: Optional[str] = None
    founder_background: Optional[str] = None
    funding_stability: Optional[str] = None
    expected_hours: Optional[str] = None

    # Candidate Targeting
    ideal_companies: Optional[List[str]] = None
    disqualifying_traits: Optional[List[str]] = None
    deal_breakers: Optional[List[str]] = None
    culture_fit_indicators: Optional[List[str]] = None
    startup_mindset_requirements: Optional[List[str]] = None
    autonomy_level_required: Optional[str] = None
    growth_mindset_indicators: Optional[List[str]] = None
    ideal_candidate_profile: Optional[str] = None

    # Interview Process
    interview_process_tags: Optional[List[str]] = None
    technical_assessment_type: Optional[List[str]] = None
    interview_focus_areas: Optional[List[str]] = None
    time_to_hire: Optional[str] = None
    decision_makers: Optional[List[str]] = None

    # Recruiter Pitch Points
    recruiter_pitch_points: Optional[List[str]] = None

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
    """Process transcript using OpenAI to extract structured information."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured"
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
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
            max_tokens=3000,
            response_format={ "type": "json_object" }
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
        # Initialize VectorStore with OpenAI support
        vector_store = VectorStore(init_openai=True)
        
        # Generate a unique job ID
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create text representation for embedding
        job_text = f"""
        Company: {job.company_name or ''}
        Role: {job.job_title or ''}
        Location: {', '.join(job.city) if job.city else ''} {', '.join(job.state) if job.state else ''}
        Tech Stack (Must Have): {', '.join(job.tech_stack_must_haves) if job.tech_stack_must_haves else ''}
        Tech Stack (Nice to Have): {', '.join(job.tech_stack_nice_to_haves) if job.tech_stack_nice_to_haves else ''}
        Experience: {job.minimum_years_of_experience or ''}
        Skills (Must Have): {', '.join(job.skills_must_have) if job.skills_must_have else ''}
        Skills (Preferred): {', '.join(job.skills_preferred) if job.skills_preferred else ''}
        Product Details: {job.product_details or ''}
        Key Responsibilities: {', '.join(job.key_responsibilities) if job.key_responsibilities else ''}
        """
        
        # Get embedding for the job text
        vector = vector_store.get_embedding(job_text)
        
        # Helper function to handle null values
        def clean_metadata_value(value):
            if value is None:
                return ""  # Convert None to empty string
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, list):
                return [str(item) for item in value] if value else []
            return str(value)
        
        # Flatten metadata for Pinecone with null handling
        metadata = {
            # Company Information
            "company_name": clean_metadata_value(job.company_name),
            "company_url": clean_metadata_value(job.company_url),
            "company_stage": clean_metadata_value(job.company_stage),
            "most_recent_funding_round_amount": clean_metadata_value(job.most_recent_funding_round_amount),
            "total_funding_amount": clean_metadata_value(job.total_funding_amount),
            "investors": clean_metadata_value(job.investors),
            "team_size": clean_metadata_value(job.team_size),
            "founding_year": clean_metadata_value(job.founding_year),
            "company_mission": clean_metadata_value(job.company_mission),
            "target_market": clean_metadata_value(job.target_market),
            "industry_vertical": clean_metadata_value(job.industry_vertical),
            "company_vision": clean_metadata_value(job.company_vision),
            "company_growth_story": clean_metadata_value(job.company_growth_story),
            "company_culture": clean_metadata_value(job.company_culture),
            "scaling_plans": clean_metadata_value(job.scaling_plans),
            "mission_and_impact": clean_metadata_value(job.mission_and_impact),
            "tech_innovation": clean_metadata_value(job.tech_innovation),
            
            # Role Details
            "job_title": clean_metadata_value(job.job_title),
            "job_url": clean_metadata_value(job.job_url),
            "positions_available": clean_metadata_value(job.positions_available),
            "hiring_urgency": clean_metadata_value(job.hiring_urgency),
            "seniority_level": clean_metadata_value(job.seniority_level),
            "work_arrangement": clean_metadata_value(job.work_arrangement),
            "cities": clean_metadata_value(job.city),
            "states": clean_metadata_value(job.state),
            "visa_sponsorship": clean_metadata_value(job.visa_sponsorship),
            "work_authorization": clean_metadata_value(job.work_authorization),
            "salary_range": clean_metadata_value(job.salary_range),
            "equity_range": clean_metadata_value(job.equity_range),
            "reporting_structure": clean_metadata_value(job.reporting_structure),
            "team_composition": clean_metadata_value(job.team_composition),
            "role_status": clean_metadata_value(job.role_status),
            
            # Technical Requirements
            "role_category": clean_metadata_value(job.role_category),
            "tech_stack_must_haves": clean_metadata_value(job.tech_stack_must_haves),
            "tech_stack_nice_to_haves": clean_metadata_value(job.tech_stack_nice_to_haves),
            "tech_stack_tags": clean_metadata_value(job.tech_stack_tags),
            "tech_breadth": clean_metadata_value(job.tech_breadth_requirement),
            "min_years_experience": clean_metadata_value(job.minimum_years_of_experience),
            "domain_expertise": clean_metadata_value(job.domain_expertise),
            "ai_ml_experience": clean_metadata_value(job.ai_ml_experience),
            "infrastructure_experience": clean_metadata_value(job.infrastructure_experience),
            "system_design_level": clean_metadata_value(job.system_design_level),
            "coding_proficiency": clean_metadata_value(job.coding_proficiency_required),
            
            # Qualification Requirements
            "leadership_requirement": clean_metadata_value(job.leadership_requirement),
            "education_requirement": clean_metadata_value(job.education_requirement),
            "advanced_degree": clean_metadata_value(job.advanced_degree_preference),
            "papers_required": clean_metadata_value(job.papers_publications_preferred),
            "startup_experience": clean_metadata_value(job.prior_startup_experience),
            "advancement_required": clean_metadata_value(job.advancement_history_required),
            "independence_level": clean_metadata_value(job.independent_work_capacity),
            "skills_must_have": clean_metadata_value(job.skills_must_have),
            "skills_preferred": clean_metadata_value(job.skills_preferred),
            
            # Product and Role Context
            "product_details": clean_metadata_value(job.product_details),
            "product_stage": clean_metadata_value(job.product_development_stage),
            "technical_challenges": clean_metadata_value(job.technical_challenges),
            "key_responsibilities": clean_metadata_value(job.key_responsibilities),
            "scope_of_impact": clean_metadata_value(job.scope_of_impact),
            "expected_deliverables": clean_metadata_value(job.expected_deliverables),
            "dev_methodology": clean_metadata_value(job.product_development_methodology),
            
            # Startup Specific Factors
            "codebase_stage": clean_metadata_value(job.stage_of_codebase),
            "growth_trajectory": clean_metadata_value(job.growth_trajectory),
            "founder_background": clean_metadata_value(job.founder_background),
            "funding_stability": clean_metadata_value(job.funding_stability),
            "expected_hours": clean_metadata_value(job.expected_hours),
            
            # Interview Process
            "interview_tags": clean_metadata_value(job.interview_process_tags),
            "assessment_type": clean_metadata_value(job.technical_assessment_type),
            "interview_focus": clean_metadata_value(job.interview_focus_areas),
            "time_to_hire": clean_metadata_value(job.time_to_hire),
            "decision_makers": clean_metadata_value(job.decision_makers),
            
            # Recruiter Pitch Points
            "recruiter_pitch_points": clean_metadata_value(job.recruiter_pitch_points),
            
            # Additional Metadata
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id
        }
        
        # Store in Pinecone with flattened metadata
        vector_store.jobs_index.upsert(vectors=[(
            job_id,
            vector,
            metadata
        )])
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Job successfully submitted and indexed"
        }
        
    except Exception as e:
        print(f"Error in submit_job: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit job: {str(e)}"
        )

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

async def retry_knowledge_base_source_cleanup(knowledge_base_id: str, source_id: str, max_retries: int = 3, delay_seconds: int = 60):
    """
    Retry deleting a knowledge base source with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"https://api.retellai.com/delete-knowledge-base-source/{knowledge_base_id}/source/{source_id}",
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code in (200, 404):
                    print(f"Successfully deleted knowledge base source {source_id} on retry attempt {attempt + 1}")
                    return True
                
            print(f"Retry attempt {attempt + 1} failed for knowledge base source {source_id}")
            
        except Exception as e:
            print(f"Error in retry attempt {attempt + 1}: {str(e)}")
    
    print(f"Failed to delete knowledge base source {source_id} after {max_retries} retries")
    return False

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
                print(f"Error from Retell client: {str(e)}")
                print(f"Error type: {type(e)}")
                print(f"Error traceback: {traceback.format_exc()}")
                return False, None
            finally:
                # Clean up the temporary file
                os.unlink(temp_file.name)
                print(f"Cleaned up temporary file: {temp_file.name}")

    except Exception as e:
        print(f"Error adding file to knowledge base: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return False, None
    finally:
        print("=== Add to knowledge base operation complete ===\n")

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
                "linkedin": linkedin,
                "source_id": source_id  # Add source_id to metadata
            },
            "retell_llm_dynamic_variables": {
                "candidate_name": name,
                "candidate_email": email,
                "candidate_linkedin": linkedin
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
        
        # If call has ended, clean up the knowledge base source
        if payload.call_status == RetellCallStatus.ENDED:
            # Try to get source_id from webhook metadata first, then fall back to our stored data
            source_id = payload.metadata.get('source_id') or call_data.get('source_id')
            
            if source_id:
                print(f"Call ended - cleaning up source {source_id}")
                knowledge_base_id = "knowledge_base_b1df2fc51182f47b"
                
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        delete_url = f"https://api.retellai.com/delete-knowledge-base-source/{knowledge_base_id}/source/{source_id}"
                        print(f"Attempting to delete source using URL: {delete_url}")
                        
                        response = await client.delete(
                            delete_url,
                            headers={
                                "Authorization": f"Bearer {RETELL_API_KEY}",
                                "Content-Type": "application/json"
                            }
                        )
                        
                        print(f"Delete response status: {response.status_code}")
                        print(f"Delete response body: {response.text}")
                        
                        if response.status_code in (200, 404):
                            print(f"Successfully deleted source {source_id}")
                            # Update our call status to reflect successful cleanup
                            call_data["source_cleaned"] = True
                        else:
                            print(f"Failed to delete source {source_id}: {response.text}")
                            # Schedule retry
                            print("Scheduling retry for source deletion")
                            asyncio.create_task(retry_knowledge_base_source_cleanup(knowledge_base_id, source_id))
                except Exception as e:
                    print(f"Error deleting source: {str(e)}")
                    print(f"Error type: {type(e)}")
                    print(f"Error traceback: {traceback.format_exc()}")
                    # Schedule retry
                    print("Scheduling retry after error")
                    asyncio.create_task(retry_knowledge_base_source_cleanup(knowledge_base_id, source_id))
            else:
                print("Warning: No source_id found in webhook metadata or stored call data")
            
        return {
            "status": "success",
            "message": "Webhook processed successfully",
            "call_id": payload.call_id,
            "call_status": payload.call_status,
            "source_id": source_id if 'source_id' in locals() else None,
            "cleanup_status": call_data.get("source_cleaned", False)
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

async def cleanup_completed_calls():
    """Background task to periodically check call statuses and cleanup completed calls."""
    print("\n=== Starting background cleanup task for completed calls ===")
    knowledge_base_id = "knowledge_base_b1df2fc51182f47b"
    
    while True:
        try:
            print("\n--- Checking for completed calls ---")
            print("\nCurrent call_statuses:")
            for call_id, data in call_statuses.items():
                print(f"Call {call_id}:")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Source ID: {data.get('source_id', 'none')}")
                print(f"  Cleaned: {data.get('source_cleaned', False)}")
                print(f"  Timestamp: {data.get('timestamp', 'none')}")
            
            # Get all calls that haven't been cleaned up yet
            active_calls = {
                call_id: data 
                for call_id, data in call_statuses.items() 
                if not data.get('source_cleaned', False)
            }
            
            if not active_calls:
                print("\nNo active calls to check")
            else:
                print(f"\nChecking {len(active_calls)} active calls")
            
            for call_id, call_data in active_calls.items():
                print(f"\nChecking call {call_id}")
                success, status, call_info = await check_call_status(call_id)
                
                if success and status == RetellCallStatus.ENDED:
                    print(f"Call {call_id} has ended - initiating cleanup")
                    source_id = call_data.get('source_id')
                    
                    if source_id:
                        print(f"Found source_id {source_id} for cleanup")
                        success, message = await delete_knowledge_base_source(
                            knowledge_base_id=knowledge_base_id,
                            source_id=source_id
                        )
                        
                        if success:
                            call_data['status'] = RetellCallStatus.ENDED
                            call_data['source_cleaned'] = True
                            call_data['cleanup_timestamp'] = datetime.utcnow().isoformat()
                            print(f"Successfully cleaned up source {source_id} for call {call_id}")
                        else:
                            print(f"Failed to clean up source {source_id}: {message}")
                    else:
                        print(f"Warning: No source_id found for call {call_id}")
                
                elif not success:
                    print(f"Failed to check status for call {call_id}: {status}")
                else:
                    print(f"Call {call_id} status: {status} - no cleanup needed yet")
            
            # Wait before next check
            print("\n--- Completed call status check cycle ---")
            await asyncio.sleep(300)  # Check every 20 seconds
            
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            await asyncio.sleep(60)  # Wait a minute before retrying if there's an error

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup."""
    try:
        # Start the cleanup task
        asyncio.create_task(cleanup_completed_calls())
        print("Started background cleanup task for completed calls")
    except Exception as e:
        print(f"Error starting background tasks: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {traceback.format_exc()}")

app = app