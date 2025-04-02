from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class CompanyStage(str, Enum):
    PRE_SEED = "Pre-Seed"
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C_PLUS = "Series C+"
    PRIVATE = "Private"
    PUBLIC = "Public"

class CompanyIndustryVertical(str, Enum):
    AI = "AI"
    HEALTHCARE = "Healthcare"
    FINTECH = "Fintech"
    INSURTECH = "Insurtech"
    ENTERPRISE_SOFTWARE = "Enterprise Software"
    DEVELOPER_TOOLS = "Developer Tools"
    CYBERSECURITY = "Cybersecurity"
    MEDIA = "Media"
    CONSTRUCTION_TECH = "Construction Tech"
    E_COMMERCE = "E-commerce"
    LOGISTICS = "Logistics"
    ROBOTICS = "Robotics"
    CLIMATE = "Climate"
    EDUCATION = "Education"
    LEGAL_TECH = "LegalTech"
    BIOTECH = "Biotech"
    IOT = "IoT"
    CONSUMER = "Consumer"
    REAL_ESTATE = "Real Estate"
    HR_TECH = "HR Tech"
    GAMING = "Gaming"
    TRAVEL = "Travel"
    SUPPLY_CHAIN = "Supply Chain"
    MANUFACTURING = "Manufacturing"

class TargetMarket(str, Enum):
    B2B = "B2B"
    B2C = "B2C"
    ENTERPRISE = "Enterprise"
    SMB = "SMB"
    CONSUMER = "Consumer"

class RoleCategory(str, Enum):
    FRONTEND = "Frontend"
    BACKEND = "Backend"
    FULL_STACK = "Full-Stack"
    INFRA = "Infra"
    DEVOPS = "DevOps"
    DATA = "Data"
    ML_AI = "ML/AI"
    MOBILE = "Mobile"
    SWE = "SWE"
    DESIGN = "Design"
    PRODUCT = "Product"
    SECURITY = "Security"
    FOUNDING_ENGINEER = "Founding Engineer"
    QA = "QA"
    EMBEDDED = "Embedded"

class Seniority(str, Enum):
    ZERO_TO_THREE = "0-3"
    THREE_TO_FIVE = "3-5"
    FIVE_TO_EIGHT = "5-8"
    EIGHT_PLUS = "8+"

class WorkArrangement(str, Enum):
    ON_SITE = "On-site"
    HYBRID = "Hybrid"
    REMOTE = "Remote"

class RoleStatus(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    CLOSED = "Closed"

class AIMLExpRequired(str, Enum):
    REQUIRED = "Required"
    PREFERRED = "Preferred"
    NOT_REQUIRED = "Not Required"

class CodingProficiency(str, Enum):
    BASIC = "Basic"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

class EducationRequired(str, Enum):
    TOP_30_CS = "Top 30 CS program"
    IVY_LEAGUE = "Ivy League"
    RESPECTED_SCHOOLS = "Respected schools"
    NO_REQUIREMENT = "No requirement"
    NO_BOOTCAMPS = "No bootcamps"

class EducationAdvancedDegree(str, Enum):
    PHD_PREFERRED = "PhD preferred"
    MASTERS_PREFERRED = "Master's preferred"
    NOT_REQUIRED = "Not required"

class AutonomyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

class JobPosting(BaseModel):
    # Basic Job Information
    job_title: str
    job_url: str
    positions_available: int
    hiring_urgency: str
    seniority: Seniority
    work_arrangement: List[WorkArrangement]
    
    # Location
    location_city: List[str]
    location_state: List[str]
    location_country: str
    visa_sponsorship: bool
    work_authorization: str
    
    # Compensation
    salary_range_min: int
    salary_range_max: int
    equity_range_min: Union[float, str] = "n/a"
    equity_range_max: Union[float, str] = "n/a"
    
    # Team Structure
    reporting_structure: str
    team_structure: str
    team_roles: List[str]
    
    # Role Details
    role_status: RoleStatus
    role_category: List[RoleCategory]
    tech_stack_must_haves: List[str]
    tech_stack_nice_to_haves: List[str]
    tech_stack_tags: List[str]
    minimum_years_of_experience: int
    domain_expertise: List[str]
    ai_ml_exp_required: AIMLExpRequired
    ai_ml_exp_focus: List[str]
    infrastructure_experience: List[str]
    system_design_expectation: str
    coding_proficiency: CodingProficiency
    languages: List[str]
    version_control: List[str]
    ci_cd_tools: List[str]
    collaboration_tools: List[str]
    leadership_required: bool
    
    # Education and Experience
    education_required: EducationRequired
    education_advanced_degree: EducationAdvancedDegree
    prior_startup_experience: bool
    startup_exp: str
    career_trajectory: str
    independent_work_capacity: AutonomyLevel
    independent_work: AutonomyLevel
    
    # Skills
    skills_must_have: List[str]
    skills_preferred: List[str]
    
    # Product Information
    product_description: str
    product_stage: str
    product_dev_methodology: List[str]
    product_technical_challenges: List[str]
    
    # Role Responsibilities
    key_responsibilities: List[str]
    scope_of_impact: List[str]
    expected_deliverables: List[str]
    
    # Company Information
    company_name: str
    company_url: str
    company_stage: CompanyStage
    company_funding_most_recent: int
    company_funding_total: int
    company_funding_investors: List[str]
    company_founded: str
    company_team_size: int
    company_mission: str
    company_vision: str
    company_growth_story: str
    company_culture: str
    company_scaling_plans: str
    company_tech_innovation: str
    company_industry_vertical: List[CompanyIndustryVertical]
    company_target_market: List[TargetMarket]
    
    # Candidate Fit
    ideal_companies: List[str]
    deal_breakers: List[str]
    disqualifying_traits: List[str]
    culture_fit: List[str]
    startup_mindset: List[str]
    autonomy_level_required: AutonomyLevel
    growth_mindset: str
    ideal_candidate_profile: str
    
    # Interview Process
    interview_process_steps: List[str]
    decision_makers: List[str]
    recruiter_pitch_points: List[str] 