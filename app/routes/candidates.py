from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, Depends
from typing import Dict, Any, Optional
import uuid
import logging
import traceback
from fastapi.responses import HTMLResponse
from app.services.candidate_service import CandidateService
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.services.matching_service import MatchingService
from app.services.job_service import JobService
from app.services.retell_service import RetellService
from app.agents.langchain.agents.candidate_intake_agent import CandidateIntakeAgent
from app.agents.langchain.agents.job_matching_agent import JobMatchingAgent
from app.agents.langchain.agents.follow_up_agent import FollowUpAgent
from app.agents.langchain.agents.interview_agent import InterviewAgent
from app.agents.langchain.agents.farming_matching_agent import FarmingMatchingAgent
from app.agents.brain_agent import BrainAgent
from app.dependencies import get_brain_agent
from app.config.settings import Settings
from app.config.supabase import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Get settings and services
settings = Settings()
supabase_client = get_supabase_client()
retell_service = RetellService()
openai_service = OpenAIService()

# Initialize services with dependencies
candidate_service = CandidateService(
    supabase_client=supabase_client,
    retell_service=retell_service,
    openai_service=openai_service,
    settings=settings
)
vector_service = VectorService()
matching_service = MatchingService()
job_service = JobService()

@router.post("/candidates")
async def create_candidate(
    background_tasks: BackgroundTasks,
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    linkedin_url: Optional[str] = Form(None),
    resume: UploadFile = File(...),
    brain_agent: BrainAgent = Depends(get_brain_agent)
):
    """
    Handle new candidate submission from frontend (multipart/form-data).
    Creates initial record, then triggers BrainAgent for further processing.
    """
    candidate_id = None
    try:
        logger.info(f"Processing form submission for {email}")

        if not resume.content_type == 'application/pdf':
             raise HTTPException(
                 status_code=400,
                 detail="Invalid file type. Resume must be a PDF."
             )

        resume_bytes = await resume.read()

        submission_data = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone,
            'linkedin_url': linkedin_url,
            'resume_content': resume_bytes,
            'resume_filename': resume.filename,
        }

        logger.info(f"Creating initial candidate record for {email}")
        initial_result = await candidate_service.create_initial_candidate(submission_data)
        candidate_id = initial_result['id']
        logger.info(f"Initial record created for candidate {candidate_id}")

        logger.info(f"Triggering BrainAgent processing for candidate {candidate_id}")
        background_tasks.add_task(
            brain_agent.handle_candidate_submission,
            candidate_id=candidate_id,
            candidate_email=email,
            resume_content=resume_bytes
        )

        logger.info(f"Successfully initiated processing for candidate {candidate_id}")
        return {"status": "success", "message": "Candidate submission received, processing started.", "candidate_id": candidate_id}

    except ValueError as e:
        logger.error(f"Validation error for {email}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing submission for candidate {candidate_id or email}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error during candidate submission.")

@router.get("/candidates/{candidate_id}/request-callback", response_class=HTMLResponse)
async def request_candidate_callback(
    candidate_id: uuid.UUID,
    brain_agent: BrainAgent = Depends(get_brain_agent)
):
    """Endpoint triggered by link in 'missed call' email to reschedule a Retell call."""
    try:
        logger.info(f"Received callback request for candidate_id: {candidate_id}")
        
        success = await brain_agent.handle_callback_request(str(candidate_id))
        
        if success:
            html_content = """
            <html>
                <head><title>Callback Requested</title></head>
                <body>
                    <h1>Thank You!</h1>
                    <p>We've received your request. Anita, our AI assistant, will try calling you again shortly.</p>
                </body>
            </html>
            """
            return HTMLResponse(content=html_content, status_code=200)
        else:
            html_content = """
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Request Failed</h1>
                    <p>Sorry, we couldn't schedule a callback at this time. Please try again later or contact support.</p>
                </body>
            </html>
            """
            return HTMLResponse(content=html_content, status_code=500)
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing callback for candidate {candidate_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        html_content = """
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Server Error</h1>
                <p>Sorry, an unexpected error occurred. Please try again later.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=500) 