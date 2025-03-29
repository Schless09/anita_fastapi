import pytest
from typing import Dict, Any
from app.services.candidate_service import CandidateService
from app.services.job_service import JobService
from app.config import get_settings, get_supabase

settings = get_settings()
supabase = get_supabase()
candidate_service = CandidateService()
job_service = JobService()

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