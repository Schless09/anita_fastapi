from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.candidate_service import CandidateService
from app.validators.candidate_validator import validate_candidate_data
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
candidate_service = CandidateService()

@router.post("/candidates")
async def create_candidate(candidate_data: Dict[str, Any]):
    """
    Handle new candidate submission from frontend.
    """
    try:
        logger.info("Validating candidate data")
        validated_data = validate_candidate_data(candidate_data)
        
        logger.info("Processing candidate submission")
        result = await candidate_service.process_candidate_submission(validated_data)
        
        logger.info(f"Successfully created candidate {result['candidate_id']}")
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating candidate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/candidates/call-complete")
async def process_call_completion(call_data: Dict[str, Any]):
    """
    Handle call completion webhook from Retell.
    """
    try:
        logger.info(f"Processing call completion for call {call_data.get('call_id')}")
        await candidate_service.process_call_completion(call_data)
        logger.info("Successfully processed call completion")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing call completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 