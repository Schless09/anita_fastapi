import logging
from fastapi import APIRouter, Depends, HTTPException, Request, Header, status
from typing import Dict, Any, Optional

from app.services.vector_service import VectorService
# Assuming you have a way to get settings, e.g., API keys or webhook secrets
# from app.config import get_settings

# settings = get_settings()
# TODO: Retrieve your webhook secret from settings/environment variables
# SUPABASE_WEBHOOK_SECRET = settings.supabase_webhook_secret
SUPABASE_WEBHOOK_SECRET = "your_secret_token" # Replace with actual secret handling

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/webhooks/jobs/inserted",
    status_code=status.HTTP_202_ACCEPTED, # Use 202 Accepted for async processing
    summary="Webhook to process new job insertions from Supabase",
    tags=["Webhooks"],
)
async def handle_new_job_webhook(
    request: Request,
    payload: Dict[str, Any],
    vector_service: VectorService = Depends(VectorService),
    # Optional: Add header for secret validation
    # authorization: Optional[str] = Header(None),
):
    """
    Receives Supabase webhook notification for new rows in jobs_dev.
    Generates embedding and metadata for the new job record.
    """
    logger.info("Received Supabase job insertion webhook.")

    # --- Optional: Security Check ---
    # if not authorization or authorization != f"Bearer {SUPABASE_WEBHOOK_SECRET}":
    #     logger.warning("Webhook received with invalid or missing secret.")
    #     raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    # logger.info("Webhook secret validated.")
    # --- End Security Check ---

    # Validate payload structure (basic check)
    if payload.get("type") != "INSERT" or "record" not in payload:
        logger.error(f"Invalid webhook payload structure: {payload}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payload structure")

    new_job_record = payload.get("record", {})
    job_id = new_job_record.get("id")

    if not job_id:
        logger.error(f"Webhook payload missing job ID in record: {new_job_record}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing job ID in record")

    logger.info(f"Processing webhook for newly inserted job ID: {job_id}")

    try:
        # Use the existing upsert_job function.
        # It will generate embedding and metadata, then UPDATE the existing row.
        # We pass the full 'new_job_record' as job_data. upsert_job already
        # handles excluding the 'id' during the PATCH/update operation.
        await vector_service.upsert_job(job_id=str(job_id), job_data=new_job_record)
        logger.info(f"Successfully processed webhook and updated job ID: {job_id}")
        return {"message": "Webhook received and job processing initiated."}

    except Exception as e:
        logger.exception(f"Error processing webhook for job ID {job_id}: {str(e)}")
        # Return 500 but Supabase might retry, consider specific error handling
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process job {job_id}") 