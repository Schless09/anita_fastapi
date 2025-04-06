from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from app.services.job_service import JobService
from app.schemas.job_posting import JobPosting
import logging
import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from app.dependencies import get_job_service

logger = logging.getLogger(__name__)
router = APIRouter()

class Location(BaseModel):
    city: List[str]
    state: List[str]
    country: str

class SalaryRange(BaseModel):
    min: str
    max: str

class Compensation(BaseModel):
    salary_range_usd: SalaryRange
    equity_range_percent: Optional[SalaryRange]

class TeamComposition(BaseModel):
    team_size: str
    structure: str
    roles: List[str]

class TechStack(BaseModel):
    must_haves: List[str]
    nice_to_haves: List[str]
    tags: List[str]

class AIMLExperience(BaseModel):
    required: bool
    preferred: bool
    focus: List[str]

class EducationRequirements(BaseModel):
    required: str
    advanced_degree_preferred: str

class Product(BaseModel):
    description: str
    development_stage: str
    technical_challenges: List[str]
    development_methodology: List[str]

class CompanyFunding(BaseModel):
    most_recent_round: str
    total: str
    investors: List[str]

class Company(BaseModel):
    name: str
    url: str
    stage: str
    funding: CompanyFunding
    team_size: str
    founded: str
    mission: str
    vision: str
    growth_story: str
    culture: str
    scaling_plans: str
    mission_and_impact: str
    tech_innovation: str

class CandidateFit(BaseModel):
    ideal_companies: List[str]
    deal_breakers: List[str]
    disqualifying_traits: List[str]
    culture_fit: List[str]
    startup_mindset: List[str]
    autonomy_level_required: str
    growth_mindset: str
    ideal_candidate_profile: str

class InterviewProcess(BaseModel):
    tags: List[str]
    assessment_type: List[str]
    focus_areas: List[str]
    time_to_hire: str
    decision_makers: List[str]

class JobSubmission(BaseModel):
    job_title: str
    job_url: str
    positions_available: str
    hiring_urgency: str
    seniority_level: str
    work_arrangement: List[str]
    location: Location
    visa_sponsorship: bool
    work_authorization: str
    compensation: Compensation
    reporting_structure: str
    team_composition: TeamComposition
    role_status: str
    role_category: str
    tech_stack: TechStack
    tech_breadth_requirement: str
    minimum_years_of_experience: str
    domain_expertise: List[str]
    ai_ml_experience: AIMLExperience
    infrastructure_experience: List[str]
    system_design_expectation: str
    coding_proficiency: str
    languages: List[str]
    version_control: List[str]
    ci_cd_tools: List[str]
    collaboration_tools: List[str]
    leadership_required: bool
    education_requirements: EducationRequirements
    prior_startup_experience: str
    advancement_history_required: bool
    independent_work_capacity: str
    skills_must_have: List[str]
    skills_preferred: List[str]
    product: Product
    key_responsibilities: List[str]
    scope_of_impact: List[str]
    expected_deliverables: List[str]
    company: Company
    candidate_fit: CandidateFit
    interview_process: InterviewProcess
    recruiter_pitch_points: List[str]

@router.post("/jobs/submit")
async def submit_job(job_data: JobPosting, job_service: JobService = Depends(get_job_service)):
    """
    Handle new job submission and generate embeddings.
    """
    try:
        # Log incoming request
        logger.info(f"Received job submission request for position: {job_data.job_title}")
        logger.info(f"Company: {job_data.company_name}")
        logger.info(f"Location: {job_data.location_country} - {', '.join(job_data.location_city)}")
        
        # Log key job details
        logger.info(f"Job Details - Seniority: {job_data.seniority}, "
                   f"Work Arrangement: {', '.join(job_data.work_arrangement)}")
        
        # Log technical requirements
        logger.info(f"Technical Requirements - Must Have: {', '.join(job_data.tech_stack_must_haves)}")
        if job_data.tech_stack_nice_to_haves:
            logger.info(f"Nice to Have: {', '.join(job_data.tech_stack_nice_to_haves)}")
        
        # Log compensation info
        logger.info(f"Salary Range: ${job_data.salary_range_min} - ${job_data.salary_range_max}")
        
        # Add job ID and timestamps
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        
        full_job_data = {
            "id": job_id,
            **job_data.model_dump(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info("Starting job processing with job service")
        try:
            result = await job_service.process_job_submission(full_job_data)
            logger.info(f"Successfully processed job submission. Job ID: {result.get('id')}")
            logger.info(f"Job Title: {result.get('job_title')}")
            logger.info(f"Company Name: {result.get('company_name')}")
            return result
        except Exception as job_service_error:
            logger.error(f"Error in job service processing: {str(job_service_error)}")
            logger.error("Job service error traceback:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing job in service: {str(job_service_error)}"
            )
        
    except ValidationError as validation_error:
        logger.error("Validation error in job submission:")
        for error in validation_error.errors():
            logger.error(f"Field: {' -> '.join(str(loc) for loc in error['loc'])}")
            logger.error(f"Error: {error['msg']}")
            logger.error(f"Input: {error.get('input', 'N/A')}")
        raise HTTPException(
            status_code=422,
            detail={"message": "Validation error", "errors": validation_error.errors()}
        )
    except Exception as e:
        error_msg = f"Unexpected error in job submission: {str(e)}"
        logger.error(error_msg)
        logger.error("Full error traceback:", exc_info=True)
        logger.error(f"Job data that caused error: {job_data.model_dump_json(indent=2)}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Internal server error", "error": str(e)}
        ) 