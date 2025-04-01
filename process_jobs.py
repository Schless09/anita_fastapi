import asyncio
import aiohttp
import json
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def submit_job(session: aiohttp.ClientSession, job_data: Dict) -> Dict:
    """Submit a single job posting to the API."""
    try:
        async with session.post(
            'http://localhost:8000/jobs/submit',
            json=job_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"Error submitting job: {error_text}")
                return {"error": error_text, "status": response.status}
    except Exception as e:
        logger.error(f"Exception submitting job: {str(e)}")
        return {"error": str(e), "status": 500}

async def process_jobs(jobs_data: List[Dict], batch_size: int = 5) -> List[Dict]:
    """Process multiple jobs in batches."""
    async with aiohttp.ClientSession() as session:
        results = []
        for i in range(0, len(jobs_data), batch_size):
            batch = jobs_data[i:i + batch_size]
            tasks = [submit_job(session, job) for job in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(jobs_data) + batch_size - 1)//batch_size}")
        return results

def load_jobs_from_file(file_path: str) -> List[Dict]:
    """Load job data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading jobs from file: {str(e)}")
        return []

def save_results(results: List[Dict], output_file: str):
    """Save processing results to a file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

async def main():
    # Example usage
    input_file = "jobs_data.json"  # Your input file containing job postings
    output_file = "processing_results.json"  # Where to save the results
    
    # Load jobs
    jobs = load_jobs_from_file(input_file)
    if not jobs:
        logger.error("No jobs loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(jobs)} jobs to process")
    
    # Process jobs
    results = await process_jobs(jobs)
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    successful = sum(1 for r in results if "error" not in r)
    logger.info(f"Processing complete. Successfully processed {successful}/{len(jobs)} jobs")

if __name__ == "__main__":
    asyncio.run(main()) 