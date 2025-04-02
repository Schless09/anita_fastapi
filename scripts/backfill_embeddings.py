import asyncio
import logging
import sys
import os
from typing import Dict, Any, List

# Add project root to sys.path to allow imports from app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.config.supabase import get_supabase_client
from app.services.vector_service import VectorService
# Attempt to import logging config, fallback to basic config if not found
try:
    from app.config.logging import setup_logging
    setup_logging()
except ImportError:
    print("Custom logging config not found (app.config.logging), using basicConfig.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

async def backfill_job_embeddings():
    """
    Fetches jobs without embeddings from the jobs_dev table
    and generates/updates their embeddings and metadata using VectorService.
    """
    supabase = None
    try:
        logger.info("Initializing Supabase client and VectorService...")
        supabase = get_supabase_client()
        vector_service = VectorService() # Uses the same Supabase client implicitly
        logger.info("Initialization complete.")

        # Fetch jobs where embedding is NULL
        # TODO: Implement pagination if there are thousands of jobs
        logger.info(f"Fetching jobs from {vector_service.jobs_table} where embedding is NULL...")
        response = await supabase.table(vector_service.jobs_table).select("*").is_("embedding", "null").execute()

        if not response.data:
            logger.info("No jobs found requiring embedding backfill.")
            return

        jobs_to_process: List[Dict[str, Any]] = response.data
        total_jobs = len(jobs_to_process)
        logger.info(f"Found {total_jobs} jobs to process.")

        processed_count = 0
        failed_count = 0

        for job in jobs_to_process:
            job_id = job.get('id')
            if not job_id:
                logger.warning(f"Skipping job due to missing ID: {job}")
                failed_count += 1
                continue

            logger.info(f"Processing job ID: {job_id} ({processed_count + failed_count + 1}/{total_jobs})")
            try:
                # Use upsert_job which handles embedding generation, metadata, and DB update
                # It will find the existing job by ID and update it.
                # Pass the entire job dictionary as job_data
                await vector_service.upsert_job(job_id=str(job_id), job_data=job)
                logger.info(f"Successfully processed job ID: {job_id}")
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process job ID {job_id}: {str(e)}", exc_info=True)
                failed_count += 1

            # Optional: Add a small delay to avoid overwhelming services/rate limits
            # await asyncio.sleep(0.1)

        logger.info("--------------------")
        logger.info("Backfill process completed.")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info("--------------------")

    except Exception as e:
        logger.exception(f"An error occurred during the backfill process: {str(e)}")
    finally:
        # Supabase Python client v1 doesn't have explicit close method
        # For v2+, you might add client closing logic here if needed.
        logger.info("Script finished.")


if __name__ == "__main__":
    # Ensure environment variables are loaded (e.g., from .env)
    # If you run this script directly, you might need to load .env explicitly
    # Example using python-dotenv (install if needed: pip install python-dotenv)
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(project_root, '.env')
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            logger.info(".env file loaded for script execution.")
        else:
            logger.warning(".env file not found, relying on environment variables for script execution.")
    except ImportError:
        logger.warning("'python-dotenv' not installed. Relying on environment variables for script execution.")

    asyncio.run(backfill_job_embeddings()) 