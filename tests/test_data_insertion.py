import pytest
from typing import Dict, Any
from app.services.candidate_service import CandidateService
from app.services.job_service import JobService
from app.config import get_settings, get_supabase_client
from app.config.settings import Settings, get_table_name
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from supabase._async.client import AsyncClient
from app.services.retell_service import RetellService

# Instantiate dependencies for testing
# It's generally better practice to use pytest fixtures for this,
# but let's fix the global instantiation first.
settings = get_settings()
supabase: AsyncClient = get_supabase_client()
# Remove old CandidateService instantiation
# candidate_service = CandidateService() # Assuming this doesn't need changes yet, or will be refactored

# Instantiate dependencies for JobService (and potentially CandidateService)
openai_service = OpenAIService(settings=settings)
candidates_table = get_table_name("candidates")
jobs_table = get_table_name("jobs")
vector_service = VectorService(
    openai_service=openai_service,
    supabase_client=supabase,
    candidates_table=candidates_table,
    jobs_table=jobs_table
)
retell_service = RetellService(settings=settings)

# Instantiate CandidateService with its dependencies
candidate_service = CandidateService(
    supabase_client=supabase,
    retell_service=retell_service,
    openai_service=openai_service,
    settings=settings
)

# Instantiate JobService with its dependencies
job_service = JobService(
    supabase_client=supabase,
    vector_service=vector_service,
    openai_service=openai_service
)

@pytest.mark.asyncio
async def test_candidate_insertion():
    """Test inserting a candidate into the system."""
    # Test data
    candidate_data = {
        "name": "Test Candidate",
        "email": "test@example.com",
        "phone": "+1234567890",
        "resume_url": "https://example.com/resume.pdf",
        "skills": ["Python", "FastAPI", "LangChain"],
        "experience_years": 5,
        "preferred_location": "Remote",
        "salary_expectation": 120000
    }
    
    # Process candidate
    result = await candidate_service.process_candidate_submission(candidate_data)
    
    # Assertions
    assert result["status"] == "success"
    assert "candidate_id" in result
    assert "profile_id" in result
    
    # Cleanup
    await supabase.table("candidates").delete().eq("id", result["candidate_id"]).execute()

@pytest.mark.asyncio
async def test_job_insertion():
    """Test inserting a job into the system."""
    # Test data
    job_data = {
        "title": "Senior Backend Engineer",
        "company": "Test Company",
        "description": "Looking for a senior backend engineer...",
        "requirements": ["Python", "FastAPI", "PostgreSQL"],
        "location": "Remote",
        "salary_range": {"min": 120000, "max": 180000},
        "employment_type": "Full-time"
    }
    
    # Process job
    result = await job_service.create_job(job_data)
    
    # Assertions
    assert result["status"] == "success"
    assert "job_id" in result
    
    # Cleanup
    await supabase.table("jobs").delete().eq("id", result["job_id"]).execute()

@pytest.mark.asyncio
async def test_matching_flow():
    """Test the complete matching flow with a candidate and job."""
    # Insert test candidate
    candidate_data = {
        "name": "Test Candidate",
        "email": "test@example.com",
        "skills": ["Python", "FastAPI", "PostgreSQL"],
        "experience_years": 5
    }
    candidate_result = await candidate_service.process_candidate_submission(candidate_data)
    
    # Insert test job
    job_data = {
        "title": "Backend Engineer",
        "requirements": ["Python", "FastAPI", "PostgreSQL"],
        "experience_required": 3
    }
    job_result = await job_service.create_job(job_data)
    
    # Test matching
    match_result = await job_service.find_matching_candidates(job_result["job_id"])
    
    # Assertions
    assert match_result["status"] == "success"
    assert len(match_result["matches"]) > 0
    assert any(m["candidate_id"] == candidate_result["candidate_id"] for m in match_result["matches"])
    
    # Cleanup
    await supabase.table("candidates").delete().eq("id", candidate_result["candidate_id"]).execute()
    await supabase.table("jobs").delete().eq("id", job_result["job_id"]).execute() 