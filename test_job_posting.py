import asyncio
import os
from app.services.job_service import JobService
from app.schemas.job_posting import (
    CompanyStage, CompanyIndustryVertical, TargetMarket, RoleCategory,
    Seniority, WorkArrangement, RoleStatus, TechBreadth, AIMLExpRequired,
    CodingProficiency, EducationRequired, EducationAdvancedDegree, AutonomyLevel
)

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["OPENAI_API_KEY"] = "test_openai_key"
os.environ["SUPABASE_URL"] = "http://localhost:54321"
os.environ["SUPABASE_KEY"] = "test_supabase_key"
os.environ["BASE_URL"] = "http://localhost:8000"

async def test_job_submission():
    # Initialize the job service
    job_service = JobService()
    
    # Sample job posting data
    job_data = {
        # Basic Job Information
        "job_title": "Senior Backend Engineer",
        "job_url": "https://example.com/jobs/senior-backend-engineer",
        "positions_available": 2,
        "hiring_urgency": "ASAP",
        "seniority": Seniority.FIVE_TO_EIGHT,
        "work_arrangement": [WorkArrangement.REMOTE],
        
        # Location
        "location_city": ["San Francisco", "New York"],
        "location_state": ["CA", "NY"],
        "location_country": "United States",
        "visa_sponsorship": False,
        "work_authorization": "Must be authorized to work in the U.S. without sponsorship",
        
        # Compensation
        "salary_range_min": 180000,
        "salary_range_max": 250000,
        "equity_range_min": "n/a",
        "equity_range_max": "n/a",
        
        # Team Structure
        "reporting_structure": "Reporting to CTO",
        "team_structure": "Backend team of 5 engineers",
        "team_roles": ["Engineers", "Product", "Design"],
        
        # Role Details
        "role_status": RoleStatus.ACTIVE,
        "role_category": RoleCategory.BACKEND,
        "tech_stack_must_haves": ["Python", "FastAPI", "PostgreSQL", "AWS"],
        "tech_stack_nice_to_haves": ["Redis", "Kubernetes"],
        "tech_stack_tags": ["Python", "FastAPI", "PostgreSQL", "AWS", "Redis", "Kubernetes"],
        "tech_breadth": TechBreadth.BACKEND,
        "tech_breadth_requirement": TechBreadth.BACKEND,
        "minimum_years_of_experience": 5,
        "domain_expertise": ["Distributed systems", "API design", "Database optimization"],
        "ai_ml_exp_required": AIMLExpRequired.PREFERRED,
        "ai_ml_exp_focus": ["LLMs", "Vector databases"],
        "infrastructure_experience": ["AWS", "Docker", "CI/CD"],
        "system_design_expectation": "Senior-level system design expected",
        "coding_proficiency": CodingProficiency.EXPERT,
        "languages": ["Python"],
        "version_control": ["Git"],
        "ci_cd_tools": ["GitHub Actions"],
        "collaboration_tools": ["Slack", "Jira"],
        "leadership_required": True,
        
        # Education and Experience
        "education_required": EducationRequired.RESPECTED_SCHOOLS,
        "education_advanced_degree": EducationAdvancedDegree.NOT_REQUIRED,
        "prior_startup_experience": True,
        "startup_exp": "Required",
        "advancement_history_required": True,
        "career_trajectory": "Required",
        "independent_work_capacity": AutonomyLevel.HIGH,
        "independent_work": AutonomyLevel.HIGH,
        
        # Skills
        "skills_must_have": [
            "Backend systems development",
            "API design and implementation",
            "Database optimization",
            "System architecture"
        ],
        "skills_preferred": [
            "ML/AI experience",
            "Team leadership",
            "Technical mentorship"
        ],
        
        # Product Information
        "product_description": "AI-powered analytics platform for business intelligence",
        "product_stage": "Scaling",
        "product_dev_methodology": ["Agile", "Customer-driven"],
        "product_technical_challenges": [
            "Scaling data processing",
            "Real-time analytics",
            "AI model deployment"
        ],
        "product_development_stage": "Scaling",
        "product_development_methodology": ["Agile", "Customer-driven"],
        
        # Role Responsibilities
        "key_responsibilities": [
            "Design and implement backend systems",
            "Lead technical decisions",
            "Mentor junior engineers",
            "Optimize system performance"
        ],
        "scope_of_impact": ["Company", "Product", "Team"],
        "expected_deliverables": [
            "Scalable backend architecture",
            "High-performance APIs",
            "Technical documentation"
        ],
        
        # Company Information
        "company_name": "TechCorp",
        "company_url": "https://techcorp.com",
        "company_stage": CompanyStage.SERIES_A,
        "company_funding_most_recent": 20000000,
        "company_funding_total": 35000000,
        "company_funding_investors": ["Sequoia", "Andreessen Horowitz"],
        "company_founded": "2020",
        "company_team_size": 50,
        "company_mission": "Empower businesses with AI-driven insights",
        "company_vision": "Be the leading AI analytics platform",
        "company_growth_story": "Growing 200% YoY, serving Fortune 500 clients",
        "company_culture": "Fast-paced, data-driven, collaborative",
        "company_scaling_plans": "Expanding engineering team",
        "company_mission_and_impact": "Transform business decision-making with AI",
        "company_tech_innovation": "AI-native analytics platform",
        "company_industry_vertical": CompanyIndustryVertical.AI,
        "company_target_market": [TargetMarket.B2B, TargetMarket.ENTERPRISE],
        
        # Candidate Fit
        "ideal_companies": ["Palantir", "Databricks", "Snowflake"],
        "deal_breakers": ["No backend experience", "No startup experience"],
        "disqualifying_traits": ["Poor communication", "No system design experience"],
        "culture_fit": ["Ownership", "Technical excellence", "Collaboration"],
        "startup_mindset": ["Bias for action", "Growth mindset"],
        "autonomy_level_required": AutonomyLevel.HIGH,
        "growth_mindset": "Curious, self-driven, comfortable with ambiguity",
        "ideal_candidate_profile": "Senior backend engineer with startup experience and strong system design skills",
        
        # Interview Process
        "interview_process_steps": ["Take-home", "System design", "Behavioral", "Offer"],
        "decision_makers": ["CTO", "CEO"],
        "recruiter_pitch_points": [
            "Join high-growth Series A startup",
            "Work with cutting-edge AI technology",
            "Competitive compensation",
            "Remote-first culture"
        ]
    }
    
    try:
        # Process the job submission
        result = await job_service.process_job_submission(job_data)
        print("Successfully processed job submission:")
        print(result)
    except Exception as e:
        print(f"Error processing job submission: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_job_submission()) 