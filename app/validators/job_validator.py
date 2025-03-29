from typing import Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime

class CompanyData(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., regex=r'^https?://(www\.)?[\w-]+(\.[\w-]+)+[/#?]?.*$')
    stage: str = Field(..., min_length=1)
    team_size: str = Field(..., min_length=1)
    founded: str = Field(..., min_length=1)
    mission: str = Field(..., min_length=1)
    vision: str = Field(..., min_length=1)
    mission_and_impact: str = Field(..., min_length=1)
    growth_story: str = Field(..., min_length=1)
    culture: str = Field(..., min_length=1)
    scaling_plans: str = Field(..., min_length=1)
    tech_innovation: str = Field(..., min_length=1)
    funding_most_recent_round: str = Field(..., min_length=1)
    funding_total: str = Field(..., min_length=1)
    funding_investors: List[str] = Field(default_factory=list)

class JobData(BaseModel):
    job_title: str = Field(..., min_length=1)
    job_url: str = Field(..., regex=r'^https?://(www\.)?[\w-]+(\.[\w-]+)+[/#?]?.*$')
    positions_available: int = Field(gt=0)
    hiring_urgency: str = Field(..., min_length=1)
    salary_min: float = Field(ge=0)
    salary_max: float = Field(ge=0)
    tech_stack_must_haves: List[str] = Field(default_factory=list)
    tech_stack_nice_to_haves: List[str] = Field(default_factory=list)
    city: List[str] = Field(default_factory=list)
    state: List[str] = Field(default_factory=list)
    country: str = Field(..., min_length=1)
    work_arrangement: List[str] = Field(default_factory=list)
    domain_expertise: List[str] = Field(default_factory=list)
    seniority_level: str = Field(..., min_length=1)
    minimum_years_of_experience: int = Field(ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('salary_max')
    def max_greater_than_min(cls, v, values):
        if 'salary_min' in values and v < values['salary_min']:
            raise ValueError('salary_max must be greater than salary_min')
        return v

    @validator('created_at', 'updated_at')
    def datetime_to_iso(cls, v):
        return v.isoformat()

class JobSubmission(BaseModel):
    company: CompanyData
    job: JobData

def validate_job_submission(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate job submission data against the schema.
    """
    try:
        validated_data = JobSubmission(**data)
        return validated_data.dict()
    except Exception as e:
        raise ValueError(f"Invalid job submission data: {str(e)}") 