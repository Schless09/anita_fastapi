import asyncio
import httpx
import json
from datetime import datetime

# Test job data with minimal required fields
test_job = {
    # Basic Job Information
    "job_title": "Senior Software Engineer",
    "job_url": "https://example.com/job/123",
    "positions_available": 1,
    "hiring_urgency": "High",
    "seniority": "5-8",
    "work_arrangement": ["Remote"],
    
    # Location
    "location_city": ["San Francisco"],
    "location_state": ["CA"],
    "location_country": "USA",
    "visa_sponsorship": False,
    "work_authorization": "US Citizen or Green Card",
    
    # Compensation
    "salary_range_min": 150000,
    "salary_range_max": 200000,
    "equity_range_min": "0.1",
    "equity_range_max": "1.0",
    
    # Team Structure
    "reporting_structure": "Reports to CTO",
    "team_structure": "Engineering Team",
    "team_roles": ["Frontend", "Backend", "DevOps"],
    
    # Role Details
    "role_status": "Active",
    "role_category": "Full-Stack",
    "tech_stack_must_haves": ["Python", "React", "AWS"],
    "tech_stack_nice_to_haves": ["TypeScript", "Docker"],
    "tech_stack_tags": ["web", "cloud"],
    "tech_breadth": "Full-Stack",
    "tech_breadth_requirement": "Full-Stack",
    "minimum_years_of_experience": 5,
    "domain_expertise": ["Web Development", "Cloud Architecture"],
    "ai_ml_exp_required": "Not Required",
    "ai_ml_exp_focus": [],
    "infrastructure_experience": ["AWS", "Docker"],
    "system_design_expectation": "High",
    "coding_proficiency": "Advanced",
    "languages": ["Python", "JavaScript"],
    "version_control": ["Git"],
    "ci_cd_tools": ["GitHub Actions"],
    "collaboration_tools": ["Slack", "Jira"],
    "leadership_required": False,
    
    # Education and Experience
    "education_required": "Respected schools",
    "education_advanced_degree": "Not required",
    "prior_startup_experience": True,
    "startup_exp": "2+ years",
    "advancement_history_required": True,
    "career_trajectory": "Growing",
    "independent_work_capacity": "High",
    "independent_work": "High",
    
    # Skills
    "skills_must_have": ["System Design", "API Development"],
    "skills_preferred": ["Team Leadership"],
    
    # Product Information
    "product_description": "Cloud-based SaaS platform",
    "product_stage": "Growth",
    "product_dev_methodology": ["Agile"],
    "product_technical_challenges": ["Scalability"],
    "product_development_stage": "Growth",
    "product_development_methodology": ["Agile"],
    
    # Role Responsibilities
    "key_responsibilities": ["Design and implement features"],
    "scope_of_impact": ["Team", "Product"],
    "expected_deliverables": ["Feature implementations"],
    
    # Company Information
    "company_name": "Test Company",
    "company_url": "https://example.com",
    "company_stage": "Series A",
    "company_funding_most_recent": 10000000,
    "company_funding_total": 15000000,
    "company_funding_investors": ["VC Firm 1"],
    "company_founded": "2020",
    "company_team_size": 50,
    "company_mission": "Build great software",
    "company_vision": "Be the best in our space",
    "company_growth_story": "Growing rapidly",
    "company_culture": "Innovative",
    "company_scaling_plans": "Aggressive growth",
    "company_mission_and_impact": "Make a difference",
    "company_tech_innovation": "Leading edge",
    "company_industry_vertical": "Enterprise Software",
    "company_target_market": ["B2B"],
    
    # Target Market
    "target_market": "B2B",
    
    # Industry Vertical
    "industry_vertical": "Enterprise Software",
    
    # Candidate Fit
    "ideal_companies": ["Tech startups"],
    "deal_breakers": ["Poor work-life balance"],
    "disqualifying_traits": ["Lack of initiative"],
    "culture_fit": ["Innovative", "Collaborative"],
    "startup_mindset": ["Growth-oriented"],
    "autonomy_level_required": "High",
    "growth_mindset": "Required",
    "ideal_candidate_profile": "Experienced full-stack developer",
    
    # Interview Process
    "interview_process_tags": ["Technical", "System Design"],
    "interview_assessment_type": ["Coding", "System Design"],
    "interview_focus_areas": ["Technical Skills", "Problem Solving"],
    "interview_time_to_hire": "2 weeks",
    "interview_decision_makers": ["CTO", "Engineering Manager"],
    "interview_process_steps": ["Initial Screen", "Technical Interview", "System Design", "Final Round"],
    "decision_makers": ["CTO", "Engineering Manager"],
    
    # Recruiter Pitch Points
    "recruiter_pitch_points": ["Competitive salary", "Remote work"]
}

async def test_job_submission():
    """Test the job submission endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            print("Submitting job...")
            response = await client.post(
                "http://127.0.0.1:8000/jobs/submit",
                json=test_job
            )
            
            print(f"Status Code: {response.status_code}")
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_job_submission()) 