from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from app.services.job_service import JobService
from app.schemas.job_posting import JobPosting
import logging
import uuid
from datetime import datetime
from pydantic import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()
job_service = JobService()

@router.post("/submit")
async def submit_job(job_data: JobPosting):
    """
    Submit a new job posting.
    """
    try:
        # Log incoming request
        logger.info(f"Received job submission request for position: {job_data.job_title}")
        logger.info(f"Company: {job_data.company_name}")
        logger.info(f"Location: {job_data.location_country} - {', '.join(job_data.location_city)}")
        
        # Add job ID and timestamps
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        
        full_job_data = {
            "id": job_id,
            **job_data.model_dump(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info("Starting job processing with job service")
        try:
            result = await job_service.process_job_submission(full_job_data)
            logger.info(f"Successfully processed job submission. Job ID: {result.get('id')}")
            logger.info(f"Job Title: {result.get('job_title')}")
            logger.info(f"Company Name: {result.get('company_name')}")
            return result
        except Exception as job_service_error:
            logger.error(f"Error in job service processing: {str(job_service_error)}")
            logger.error("Job service error traceback:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing job in service: {str(job_service_error)}"
            )
        
    except ValidationError as validation_error:
        logger.error("Validation error in job submission:")
        for error in validation_error.errors():
            logger.error(f"Field: {' -> '.join(str(loc) for loc in error['loc'])}")
            logger.error(f"Error: {error['msg']}")
            logger.error(f"Input: {error.get('input', 'N/A')}")
        raise HTTPException(
            status_code=422,
            detail={"message": "Validation error", "errors": validation_error.errors()}
        )
    except Exception as e:
        error_msg = f"Unexpected error in job submission: {str(e)}"
        logger.error(error_msg)
        logger.error("Full error traceback:", exc_info=True)
        logger.error(f"Job data that caused error: {job_data.model_dump_json(indent=2)}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Internal server error", "error": str(e)}
        ) 