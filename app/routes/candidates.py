from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
from typing import Dict, Any, Optional
from app.services.candidate_service import CandidateService
import logging
from app.agents.brain_agent import BrainAgent

logger = logging.getLogger(__name__)
router = APIRouter()
candidate_service = CandidateService()
brain_agent = BrainAgent()

@router.post("/candidates")
async def create_candidate(
    background_tasks: BackgroundTasks,
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    linkedin_url: Optional[str] = Form(None),
    resume: UploadFile = File(...)
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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error during candidate submission.") 