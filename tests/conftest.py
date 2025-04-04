import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any, Generator
import asyncio
from unittest.mock import Mock, AsyncMock

from app.main import app
from app.config import get_settings
from app.services.candidate_service import CandidateService
from app.services.job_service import JobService
from app.agents.brain_agent import BrainAgent

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client() -> Generator:
    """Create a test client for FastAPI app."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    mock = AsyncMock()
    mock.table.return_value.insert.return_value.execute.return_value = {"data": [{"id": "test-id"}]}
    mock.table.return_value.select.return_value.execute.return_value = {"data": [{"id": "test-id"}]}
    return mock

@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client."""
    mock = AsyncMock()
    mock.embeddings.create.return_value = {
        "data": [{"embedding": [0.1] * 1536}]
    }
    return mock

@pytest.fixture
def test_candidate_data() -> Dict[str, Any]:
    """Sample candidate data for testing."""
    return {
        "name": "Test Candidate",
        "email": "test@example.com",
        "phone": "+1234567890",
        "resume_url": "https://example.com/resume.pdf",
        "skills": ["Python", "FastAPI", "LangChain"],
        "experience_years": 5,
        "preferred_location": "Remote",
        "salary_expectation": 120000
    }

@pytest.fixture
def test_job_data() -> Dict[str, Any]:
    """Sample job data for testing."""
    return {
        "title": "Senior Backend Engineer",
        "company": "Test Company",
        "description": "Looking for a senior backend engineer...",
        "requirements": ["Python", "FastAPI", "PostgreSQL"],
        "location": "Remote",
        "salary_range": {"min": 120000, "max": 180000},
        "employment_type": "Full-time"
    }

@pytest.fixture
def mock_brain_agent(mock_supabase, mock_openai):
    """Create a mock BrainAgent with mocked dependencies."""
    agent = BrainAgent()
    agent.candidate_service = AsyncMock()
    agent.job_service = AsyncMock()
    return agent 