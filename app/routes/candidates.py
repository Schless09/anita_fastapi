from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, Depends, Request
from typing import Dict, Any, Optional, List
import uuid
import logging
import traceback
from fastapi.responses import HTMLResponse
from app.services.candidate_service import CandidateService
from app.services.storage_service import StorageService
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
from app.dependencies import (
    get_brain_agent,
    get_candidate_service,
    get_job_service,
    get_matching_service,
    get_vector_service,
    get_openai_service,
    get_retell_service,
    get_storage_service
)
from app.config.settings import Settings
from app.config.supabase import get_supabase_client
from app.config.utils import get_table_name
from app.schemas.candidate import (
    WorkEnvironmentEnum,
    WorkAuthorizationEnum,
    VisaTypeEnum,
    EmploymentTypeEnum,
    AvailabilityEnum
)
import json

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/candidates", tags=["Candidates"])
async def create_candidate(
    request: Request,
    background_tasks: BackgroundTasks,
    firstName: str = Form(..., alias="firstName"),
    lastName: str = Form(..., alias="lastName"),
    email: str = Form(...),
    phone: str = Form(...),
    linkedinURL: Optional[str] = Form(None, alias="linkedinURL"),
    resume: UploadFile = File(...),
    workEnvironment: List[WorkEnvironmentEnum] = Form(..., alias="workEnvironment"),
    workAuthorization: WorkAuthorizationEnum = Form(..., alias="workAuthorization"),
    visaType: Optional[VisaTypeEnum] = Form(None, alias="visaType"),
    employmentType: List[EmploymentTypeEnum] = Form(..., alias="employmentType"),
    availability: str = Form(..., alias="availability"),
    dreamRoleDescription: Optional[str] = Form(None, alias="dreamRoleDescription"),
    smsConsent: bool = Form(..., alias="smsConsent"),
    legalConsent: bool = Form(..., alias="legalConsent"),
    brain_agent: BrainAgent = Depends(get_brain_agent),
    candidate_service: CandidateService = Depends(get_candidate_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Handle new candidate submission from frontend (multipart/form-data).
    Relies on FastAPI Form parameters for validation and type conversion.
    """
    logger.info("--- Entering create_candidate ---")
    
    # --- Keep List Extraction Logic (from both branches, seems consistent) --- 
    try:
        raw_form_data = await request.form() # Still need raw form for lists 
        actual_desired_locations = raw_form_data.getlist("desiredLocation")
        sub_locations_data = raw_form_data.get("subLocationsData")
        if sub_locations_data:
            try: actual_pref_sub_locations = json.loads(sub_locations_data)
            except json.JSONDecodeError: actual_pref_sub_locations = []
        else: actual_pref_sub_locations = []
        actual_work_environments = raw_form_data.getlist("workEnvironment")
        actual_employment_types = raw_form_data.getlist("employmentType")
        
        # --- Process Extracted Lists --- 
        valid_work_envs = []
        for env in actual_work_environments:
            try: valid_work_envs.append(WorkEnvironmentEnum(env))
            except ValueError: logger.warning(f"Invalid workEnvironment value: {env}")
        
        valid_emp_types = []
        for emp_type in actual_employment_types:
             try: valid_emp_types.append(EmploymentTypeEnum(emp_type))
             except ValueError: logger.warning(f"Invalid employmentType value: {emp_type}")

    except Exception as form_read_err:
        logger.error(f"Error manually reading list form data: {form_read_err}")
        raise HTTPException(status_code=400, detail="Could not process form data.")
    # --- End List Extraction --- 

    candidate_id = None # Initialize candidate_id
    try:
        logger.info(f"Processing form submission for {email} via /candidates")
        full_name = f"{firstName} {lastName}"

        # --- Convert availability string to int --- 
        try:
            availability_int = int(availability) 
            # Optional: Add validation for the range (2-5)
            if not 2 <= availability_int <= 5:
                 logger.warning(f"Received invalid availability score: {availability}. Defaulting to null/None.")
                 availability_int = None # Or handle as an error
        except ValueError:
            logger.error(f"Could not convert availability '{availability}' to integer. Setting to None.")
            availability_int = None # Set to None or raise an error if it's mandatory
        # -------------------------------------------

        if not resume.content_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
             logger.warning(f"Invalid resume file type received: {resume.content_type}")
             if not resume.content_type == 'application/pdf':
                 raise HTTPException(
                     status_code=400,
                     detail="Invalid file type. Resume must be a PDF for current processing."
                 )

        resume_bytes = await resume.read()

        # --- Keep matching-improvement version for submission_data --- 
        submission_data = {
            'name': full_name,
            'email': email,
            'phone_number': phone,
            'linkedin': linkedinURL,
            'work_environment': valid_work_envs if valid_work_envs else None,
            'desired_locations': actual_desired_locations if actual_desired_locations else None,
            'preferred_sub_locations': actual_pref_sub_locations if actual_pref_sub_locations else None,
            'work_authorization': workAuthorization,
            'visa_type': visaType,
            'employment_types': valid_emp_types if valid_emp_types else None,
            'availability': availability_int,
            'dream_role_description': dreamRoleDescription,
            'sms_consent': smsConsent, # Use boolean from Form param
            'legal_consent': legalConsent # Use boolean from Form param
        }
        # --- End matching-improvement version --- 

        logger.info(f"Creating initial candidate record for {email} with detailed form data.")
        # Use candidate_service dependency
        initial_result = await candidate_service.create_initial_candidate(submission_data)
        candidate_id = initial_result['id']
        logger.info(f"Initial record created for candidate {candidate_id}")

        resume_path = None
        try:
            logger.info(f"Storing resume file for candidate {candidate_id}")
            # Use storage_service dependency
            resume_path = await storage_service.store_resume(
                file_content=resume_bytes,
                user_id=str(candidate_id),
                original_filename=resume.filename
            )
            logger.info(f"Resume stored successfully at {resume_path}")

            await candidate_service.update_resume_path(candidate_id, resume_path)
            logger.info(f"Updated resume path for candidate {candidate_id}")

        except Exception as e:
            logger.error(f"Failed to store or update resume path for {candidate_id}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store resume file for candidate {candidate_id}"
            )

        logger.info(f"Triggering BrainAgent processing for candidate {candidate_id}")
        # Use background_tasks and brain_agent dependency
        background_tasks.add_task(
            brain_agent.handle_candidate_submission,
            candidate_id=str(candidate_id),
            candidate_email=email,
            resume_content=resume_bytes
        )

        logger.info(f"Successfully initiated processing for candidate {candidate_id}")
        # Restore original success response
        return {
            "status": "success",
            "message": "Candidate submission received, processing started.",
            "candidate_id": str(candidate_id),
            "resume_path": resume_path
        }

    # Restore original error handling
    except ValueError as e:
        logger.error(f"Validation error processing submission for {email}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing submission for candidate {candidate_id or email}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during candidate submission.")

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