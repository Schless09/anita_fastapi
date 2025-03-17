from fastapi import FastAPI, HTTPException, status
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
RETELL_AGENT_ID = os.getenv('AGENT_ID')

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
        return v.strip()

    @validator('call_id')
    def validate_call_id(cls, v):
        if not v or not v.strip():
            raise ValueError("call_id cannot be empty")
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
    resume: Optional[str] = None

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
        phone = '+' + ''.join(filter(str.isdigit, v.replace('+', '')))
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

    @validator('resume')
    def validate_resume(cls, v):
        if not v:
            return None
        # Truncate if too long
        max_length = 6000
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

class MakeCallRequest(BaseModel):
    name: str
    email: str
    phone_number: str
    linkedin: Optional[str] = None

@app.get("/")
async def read_root():
    """API root endpoint"""
    return {
        "name": "Anita AI Recruitment API",
        "version": "2.0.0",
        "status": "operational"
    }

@app.post("/candidates", response_model=Dict[str, Any])
async def submit_candidate(candidate: CandidateData):
    """Submit a new candidate"""
    try:
        # Convert candidate model to dict
        candidate_dict = candidate.dict()
        
        # Generate a unique candidate ID
        candidate_id = f"candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Store candidate in vector database
        result = brain_agent.handle_candidate_submission(candidate_dict)
        
        if result.get('status') != 'success':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to store candidate",
                    "message": result.get('message', 'Unknown error occurred')
                }
            )
            
        return {
            "status": "success",
            "message": "Candidate submitted successfully",
            "candidate_id": candidate_id,
            "data": candidate_dict
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Validation error", "message": str(e)}
        )
    except Exception as e:
        print(f"Error in submit_candidate: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)}
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

@app.post("/candidate/retell-transcript", response_model=Dict[str, Any])
async def fetch_and_store_retell_transcript(call_data: RetellCallData):
    """
    Fetch transcript from Retell AI call and store it in the candidate's profile.
    
    This endpoint will:
    1. Fetch the call transcript from Retell AI
    2. Extract relevant information from the call
    3. Update the candidate's vector store entry with the transcript
    4. Return the updated candidate profile
    
    The transcript will be used to enhance matching accuracy and candidate assessment.
    
    Raises:
        400 - Bad Request: Invalid input data
        401 - Unauthorized: Invalid Retell AI API key
        404 - Not Found: Call not found
        422 - Unprocessable Entity: Invalid call status or missing transcript
        500 - Internal Server Error: Server-side errors
        503 - Service Unavailable: Retell AI service unavailable
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
    
    # Initialize candidate profile if it doesn't exist
    if call_data.candidate_id not in brain_agent.candidate_profiles:
        brain_agent.candidate_profiles[call_data.candidate_id] = {
            'basic_info': {},
            'transcript': None,
            'processed_transcript': None,
            'screening_result': None,
            'match_result': None,
            'vector_id': None,
            'dealbreakers': None,
            'match_reason': None
        }
    
    try:
        # Fetch call transcript from Retell AI
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{RETELL_API_BASE}/get-call/{call_data.call_id}",
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}"
                    }
                )
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "Timeout",
                        "message": "Request to Retell AI timed out",
                        "action_required": "Please try again later"
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
            
            retell_data = response.json()
            
            # Validate Retell response
            try:
                validated_data = await validate_retell_response(retell_data)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "Invalid call data",
                        "message": str(e),
                        "action_required": "Please ensure the call is completed"
                    }
                )
            
            # Extract transcript and analysis
            transcript = validated_data.get('transcript', '')
            call_analysis = validated_data.get('call_analysis', {})
            
            # Process transcript with OpenAI
            processed_data = await process_transcript_with_openai(transcript)
            
            # Calculate call duration if timestamps available
            call_duration = None
            start_time = validated_data.get('start_timestamp')
            end_time = validated_data.get('end_timestamp')
            if start_time and end_time:
                call_duration = (end_time - start_time) / 1000  # Convert to seconds
            
            # Create enhanced transcript with analysis
            enhanced_transcript = EnhancedTranscript(
                raw_transcript=transcript,
                call_summary=call_analysis.get('call_summary', ''),
                user_sentiment=call_analysis.get('user_sentiment', 'Unknown'),
                call_successful=call_analysis.get('call_successful', False),
                custom_analysis=call_analysis.get('custom_analysis_data', {}),
                timestamp=datetime.utcnow(),
                call_status=validated_data['call_status'],
                call_duration=call_duration,
                error_details=validated_data.get('disconnection_reason')
            )
            
            # Update candidate profile with transcript and processed data
            result = brain_agent.add_transcript_to_profile(
                call_data.candidate_id,
                {
                    "raw_transcript": transcript,
                    "processed_data": processed_data,
                    "enhanced_transcript": enhanced_transcript.dict()
                }
            )
            
            if result.get('status') == 'error':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Profile update failed",
                        "message": result['message'],
                        "action_required": "Please check the candidate profile data"
                    }
                )
            
            return {
                'status': 'success',
                'message': 'Transcript fetched and stored successfully',
                'transcript_data': enhanced_transcript.dict(),
                'candidate_state': result.get('current_state'),
                'call_id': call_data.call_id,
                'metadata': {
                    'processed_at': datetime.utcnow().isoformat(),
                    'call_duration_seconds': call_duration,
                    'sentiment': enhanced_transcript.user_sentiment,
                    'success_status': enhanced_transcript.call_successful
                }
            }
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Communication error",
                "message": f"Failed to communicate with Retell AI: {str(e)}",
                "action_required": "Please check your network connection and try again"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "action_required": "Please contact support if the issue persists"
            }
        )

@app.post("/api/process-retell-transcript", response_model=Dict[str, Any])
async def process_retell_transcript_alias(call_data: RetellCallData):
    """Alias endpoint for /candidate/retell-transcript to maintain compatibility with frontend."""
    return await fetch_and_store_retell_transcript(call_data)

@app.post("/calls/list", response_model=RetellCallList)
async def list_retell_calls(page: int = 1, page_size: int = 10):
    """
    List all call transcripts from Retell AI.
    
    Parameters:
        page: Page number (1-based indexing)
        page_size: Number of calls per page
        
    Returns:
        List of calls with their transcripts and metadata
        
    Raises:
        401 - Unauthorized: Invalid Retell AI API key
        500 - Internal Server Error: Server-side errors
        503 - Service Unavailable: Retell AI service unavailable
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{RETELL_API_BASE}/list-calls",
                    json={
                        "page": page,
                        "page_size": page_size
                    },
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "Timeout",
                        "message": "Request to Retell AI timed out",
                        "action_required": "Please try again later"
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
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "Retell AI service error",
                        "message": f"Failed to fetch calls: {response.text}",
                        "action_required": "Please try again later"
                    }
                )
            
            try:
                data = response.json()
                if isinstance(data, list):
                    # If response is a list of calls
                    return RetellCallList(
                        calls=data,
                        total_count=len(data),
                        page=page,
                        page_size=page_size
                    )
                else:
                    # If response is an object with calls array
                    calls = data.get('calls', [])
                    if not isinstance(calls, list):
                        raise ValueError("Invalid response format: 'calls' is not a list")
                        
                    return RetellCallList(
                        calls=calls,
                        total_count=data.get('total_count', len(calls)),
                        page=page,
                        page_size=page_size
                    )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "Response format error",
                        "message": str(e),
                        "action_required": "Please contact support"
                    }
                )
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Communication error",
                "message": f"Failed to communicate with Retell AI: {str(e)}",
                "action_required": "Please check your network connection and try again"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "action_required": "Please contact support if the issue persists"
            }
        )

@app.post("/api/makeCall", response_model=Dict[str, Any])
async def make_call(request: MakeCallRequest):
    """
    Create a new Retell AI call for a candidate.
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
    
    if not RETELL_AGENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Configuration error",
                "message": "Retell AI Agent ID not configured",
                "action_required": "Please set AGENT_ID in environment variables"
            }
        )
    
    try:
        # Generate a unique candidate ID
        candidate_id = f"candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Format phone number to E.164 format
        phone = request.phone_number.strip()
        # Remove any non-digit characters except '+'
        phone = '+' + ''.join(filter(str.isdigit, phone.replace('+', '')))
        if not phone.startswith('+'):
            phone = '+' + phone
        
        # Store candidate data first
        candidate_data = {
            "name": request.name,
            "email": request.email,
            "phone_number": phone,
            "linkedin": request.linkedin
        }
        
        # Store in vector database
        try:
            vector_result = vector_store.store_candidate(candidate_id, candidate_data)
            if vector_result.get('status') != 'success':
                print(f"Warning: Failed to store candidate data: {vector_result.get('message')}")
        except Exception as e:
            print(f"Warning: Error storing candidate data: {str(e)}")
        
        # Create a new call with Retell AI
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                retell_payload = {
                    "to_number": phone,
                    "from_number": RETELL_FROM_NUMBER,
                    "agent_id": RETELL_AGENT_ID,
                    "metadata": {
                        "candidate_id": candidate_id,
                        "name": request.name,
                        "email": request.email,
                        "linkedin": request.linkedin or "",
                        "source": "anita_ai"
                    }
                }
                
                print(f"Sending request to Retell AI: {retell_payload}")
                response = await client.post(
                    f"{RETELL_API_BASE}/create-phone-call",
                    headers={
                        "Authorization": f"Bearer {RETELL_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=retell_payload
                )
                
                print(f"Retell API Response Status: {response.status_code}")
                print(f"Retell API Response Body: {response.text}")
                
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "Timeout",
                        "message": "Request to Retell AI timed out",
                        "action_required": "Please try again later"
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
            elif response.status_code != 200:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        error_msg = error_data.get('message', error_data.get('error', response.text))
                except:
                    pass
                
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "Retell AI service error",
                        "message": f"Failed to create call: {error_msg}",
                        "action_required": "Please verify the phone number and try again"
                    }
                )
            
            call_data = response.json()
            
            return {
                "status": "success",
                "message": "Call created successfully",
                "candidate_id": candidate_id,  # Return the generated candidate ID
                "call_id": call_data.get("callId") or call_data.get("call_id"),
                "call_details": {
                    "status": call_data.get("status"),
                    "created_at": datetime.utcnow().isoformat(),
                    "phone_number": phone,
                    "name": request.name,
                    "email": request.email
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in make_call: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "action_required": "Please contact support if the issue persists"
            }
        )

