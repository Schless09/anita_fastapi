from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.job_service import JobService
from app.validators.job_validator import validate_job_submission
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
job_service = JobService()

@router.post("/jobs")
async def create_job(job_data: Dict[str, Any]):
    """
    Handle new job submission.
    """
    try:
        logger.info("Validating job submission data")
        validated_data = validate_job_submission(job_data)
        
        logger.info("Processing job submission")
        result = await job_service.process_job_submission(validated_data)
        
        logger.info(f"Successfully created job {result['job_id']} for company {result['company_id']}")
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 