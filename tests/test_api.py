import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

pytestmark = pytest.mark.asyncio

async def test_health_check(test_client: TestClient):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

async def test_create_candidate(
    test_client: TestClient,
    test_candidate_data,
    mock_brain_agent
):
    """Test creating a new candidate."""
    mock_brain_agent.handle_candidate_submission.return_value = {
        "status": "success",
        "candidate_id": "test-id",
        "profile_id": "profile-id"
    }
    
    response = test_client.post("/candidates", json=test_candidate_data)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "candidate_id" in response.json()
    assert "profile_id" in response.json()

async def test_get_candidate(
    test_client: TestClient,
    mock_candidate_service
):
    """Test retrieving a candidate."""
    candidate_id = "test-id"
    mock_candidate_service.get_candidate.return_value = {
        "id": candidate_id,
        "name": "Test Candidate",
        "email": "test@example.com"
    }
    
    response = test_client.get(f"/candidates/{candidate_id}")
    
    assert response.status_code == 200
    assert response.json()["id"] == candidate_id

async def test_create_job(
    test_client: TestClient,
    test_job_data,
    mock_brain_agent
):
    """Test creating a new job."""
    mock_brain_agent.handle_job_matching.return_value = {
        "status": "success",
        "job_id": "test-job-id"
    }
    
    response = test_client.post("/jobs", json=test_job_data)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "job_id" in response.json()

async def test_get_job_matches(
    test_client: TestClient,
    mock_brain_agent
):
    """Test getting matches for a job."""
    job_id = "test-job-id"
    mock_brain_agent.handle_job_matching.return_value = {
        "status": "success",
        "matches": [
            {"candidate_id": "candidate-1", "score": 0.9},
            {"candidate_id": "candidate-2", "score": 0.8}
        ]
    }
    
    response = test_client.get(f"/jobs/{job_id}/matches")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["matches"]) == 2

async def test_get_candidate_matches(
    test_client: TestClient,
    mock_brain_agent
):
    """Test getting matches for a candidate."""
    candidate_id = "test-candidate-id"
    mock_brain_agent.handle_farming_matching.return_value = {
        "status": "success",
        "matches": [
            {"job_id": "job-1", "score": 0.9},
            {"job_id": "job-2", "score": 0.8}
        ]
    }
    
    response = test_client.get(f"/candidates/{candidate_id}/matches")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["matches"]) == 2

async def test_schedule_interview(
    test_client: TestClient,
    mock_brain_agent
):
    """Test scheduling an interview."""
    mock_brain_agent.handle_interview_scheduling.return_value = {
        "status": "success",
        "interview_id": "interview-1",
        "scheduled_time": "2024-03-20T10:00:00Z"
    }
    
    response = test_client.post("/interviews/schedule", params={
        "candidate_id": "test-candidate-id",
        "job_id": "test-job-id"
    })
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "interview_id" in response.json()
    assert "scheduled_time" in response.json()

async def test_error_handling(test_client: TestClient, mock_brain_agent):
    """Test error handling in endpoints."""
    mock_brain_agent.handle_candidate_submission.side_effect = Exception("Test error")
    
    response = test_client.post("/candidates", json={})
    
    assert response.status_code == 500
    assert "detail" in response.json()

@pytest.fixture
def mock_candidate_service():
    """Create a mock candidate service."""
    return AsyncMock() 