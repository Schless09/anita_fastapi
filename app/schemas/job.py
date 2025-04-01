from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated, Union
from datetime import datetime
from uuid import UUID
from enum import Enum

# Import all the enum types from the database
from app.db.enums import (
    CompanyStage,
    CompanyIndustryVertical,
    TargetMarket,
    RoleCategory,
    Seniority,
    WorkArrangement,
    RoleStatus,
    AutonomyLevel,
    CodingProficiency,
    TechBreadth,
    AIMLExpRequired,
    ProductStage,
    DevMethodology,
    EducationRequired,
    EducationAdvancedDegree
)

class Location(BaseModel):
    city: List[str]
    state: List[str]
    country: str

class SalaryRange(BaseModel):
    min: Union[str, int]  # Accept both string and integer values
    max: Union[str, int]  # Accept both string and integer values

class Compensation(BaseModel):
    salary_range_usd: SalaryRange
    equity_range_percent: SalaryRange

class TeamComposition(BaseModel):
    team_size: str
    structure: str
    roles: List[str]

class TechStack(BaseModel):
    must_haves: List[str]
    nice_to_haves: List[str]
    tags: Optional[List[str]] = []  # Made optional with default empty list

class AIMLExperience(BaseModel):
    required: AIMLExpRequired
    focus: List[str]

class Product(BaseModel):
    description: str
    stage: ProductStage
    dev_methodology: List[DevMethodology]
    technical_challenges: List[str]
    development_stage: Optional[ProductStage] = None  # Made optional
    development_methodology: Optional[List[DevMethodology]] = None  # Made optional and changed to list

class CompanyFunding(BaseModel):
    most_recent_round: Union[str, int]  # Accept both string and integer values
    total: Union[str, int]  # Accept both string and integer values
    investors: List[str]

class Company(BaseModel):
    name: str
    url: str
    stage: CompanyStage
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
    industry_vertical: Optional[CompanyIndustryVertical] = None  # Added optional field
    target_market: Optional[List[TargetMarket]] = None  # Added optional field

class CandidateFit(BaseModel):
    ideal_companies: List[str]
    deal_breakers: List[str]
    disqualifying_traits: List[str]
    culture_fit: List[str]
    startup_mindset: List[str]
    autonomy_level_required: AutonomyLevel
    growth_mindset: Union[str, List[str]]  # Accept both string and list values
    ideal_candidate_profile: Optional[str] = None  # Made optional

class InterviewProcess(BaseModel):
    description: str
    steps: Optional[List[str]] = None  # Made optional
    duration: Optional[str] = None  # Made optional
    format: Optional[str] = None  # Made optional
    tags: List[str] = []  # Added required field with default empty list
    assessment_type: Optional[str] = None  # Made optional
    focus_areas: List[str] = []  # Added required field with default empty list
    time_to_hire: Optional[str] = None  # Made optional
    decision_makers: List[str] = []  # Added required field with default empty list

    @classmethod
    def from_string(cls, text: str) -> "InterviewProcess":
        """Create an InterviewProcess instance from a string description."""
        return cls(
            description=text,
            tags=[],
            focus_areas=[],
            decision_makers=[]
        )

class JobBase(BaseModel):
    title: str
    company: str
    description: str
    requirements: Annotated[List[str], Field(min_items=1)]
    location: str
    salary_range: Optional[SalaryRange] = None
    employment_type: str

class JobCreate(BaseModel):
    job_title: str
    job_url: str
    positions_available: str
    hiring_urgency: str
    seniority: Seniority
    seniority_level: Optional[Seniority] = None  # Made optional
    work_arrangement: List[WorkArrangement]
    location: Location
    visa_sponsorship: bool
    work_authorization: str
    compensation: Compensation
    reporting_structure: str
    team_composition: TeamComposition
    role_status: RoleStatus
    role_category: RoleCategory
    tech_stack: TechStack
    tech_breadth: TechBreadth
    tech_breadth_requirement: Optional[TechBreadth] = None  # Made optional
    minimum_years_of_experience: Union[str, int]  # Accept both string and integer values
    domain_expertise: List[str]
    ai_ml_experience: AIMLExperience  # Changed to AIMLExperience type
    ai_ml_exp: AIMLExperience
    infrastructure_experience: List[str]
    system_design_expectation: str
    coding_proficiency: CodingProficiency
    languages: List[str]
    version_control: List[str]
    ci_cd_tools: List[str]
    collaboration_tools: List[str]
    leadership_required: bool
    education_requirements: Dict[str, str]  # Changed to Dict type
    education: Dict[str, str]
    prior_startup_experience: bool  # Changed to bool type
    startup_exp: AIMLExpRequired
    advancement_history_required: bool  # Changed to bool type
    career_trajectory: AIMLExpRequired
    independent_work_capacity: Optional[AutonomyLevel] = None  # Made optional
    independent_work: AutonomyLevel
    skills_must_have: List[str]
    skills_preferred: List[str]
    product: Product
    key_responsibilities: List[str]
    scope_of_impact: List[str]
    expected_deliverables: List[str]
    company: Company
    candidate_fit: CandidateFit
    interview_process: Union[str, InterviewProcess]  # Allow both string and object
    recruiter_pitch_points: List[str]

class JobUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[List[str]] = None
    location: Optional[str] = None
    salary_range: Optional[SalaryRange] = None
    employment_type: Optional[str] = None

class JobResponse(JobBase):
    id: str
    created_at: datetime
    updated_at: datetime
    status: str
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True 