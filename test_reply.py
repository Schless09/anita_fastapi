from agents.interaction_agent import InteractionAgent
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Test data (same as test_email.py)
test_candidate_id = "test_harrison"
test_candidate_profile = {
    "processed_data": {
        "candidate_name": "Harrison Franke",
        "contact_information": "harrison.franke@gmail.com",
        "current_role": "Software Engineer",
        "tech_stack": ["Python", "FastAPI", "React", "TypeScript"],
        "years_of_experience": 5,
        "preferred_locations": [{"city": "San Francisco", "state": "CA"}],
        "salary_expectations": {"min": 150000, "max": 200000},
        "career_goals": ["Technical Leadership", "AI/ML Development"],
        "preferred_work_environment": "Remote"
    }
}

test_job = {
    "id": "test_job_1",
    "title": "Senior Software Engineer",
    "company": "TechCorp AI",
    "location": "San Francisco",
    "tech_stack": ["Python", "FastAPI", "AI/ML", "Cloud"],
    "salary_range": {"min": 160000, "max": 220000},
    "company_stage": "Series B",
    "description": "Looking for a senior engineer to help build our next-generation AI platform.",
    "role_details": {
        "city": ["San Francisco"],
        "state": ["CA"],
        "work_authorization": "US Citizen/Green Card",
        "salary_range": "160k-220k",
        "requirements": ["5+ years of experience", "Strong Python skills", "AI/ML experience"],
        "benefits": ["Competitive salary", "Health insurance", "401k matching", "Remote work options"],
        "work_arrangement": "Hybrid (2 days in office)",
        "interview_process": "1. Initial call\n2. Technical assessment\n3. Team interview\n4. Final round",
        "team": "AI Platform Team",
        "responsibilities": ["Design and implement AI/ML systems", "Lead technical initiatives", "Mentor junior engineers"]
    }
}

def process_reply():
    # Initialize interaction agent
    interaction_agent = InteractionAgent()
    interaction_agent.vector_store = None  # Bypass vector store initialization
    
    # Update agent with candidate knowledge
    interaction_agent.update_candidate_knowledge(test_candidate_id, test_candidate_profile)
    
    # Store job data in conversation history
    interaction_agent.conversation_history[test_job["id"]] = test_job
    
    # Simulate the reply email content
    reply_content = "So what would my main responsibilities be?"
    
    # Process the reply
    result = interaction_agent.handle_candidate_reply(test_candidate_id, reply_content, test_job["id"])
    print(f"Reply processing result: {result}")

if __name__ == "__main__":
    process_reply() 