from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class WorkPreferences(BaseModel):
    arrangement: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)

class SalaryExpectations(BaseModel):
    min: float = Field(ge=0)
    max: float = Field(ge=0)
    currency: str = Field(default="USD")

    @validator('max')
    def max_greater_than_min(cls, v, values):
        if 'min' in values and v < values['min']:
            raise ValueError('max must be greater than min')
        return v

class ProfileJson(BaseModel):
    current_role: str = Field(..., min_length=1)
    current_company: str = Field(..., min_length=1)
    years_of_experience: float = Field(ge=0)
    tech_stack: List[str] = Field(default_factory=list)
    previous_companies: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    career_goals: str = Field(..., min_length=1)
    salary_expectations: SalaryExpectations
    work_preferences: WorkPreferences
    industry_preferences: List[str] = Field(default_factory=list)

class CandidateData(BaseModel):
    full_name: str = Field(..., min_length=1)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    phone: str = Field(..., pattern=r'^\+?1?\d{9,15}$')
    linkedin_url: str = Field(..., pattern=r'^https://(www\.)?linkedin\.com/in/[\w-]+/?$')
    github_url: str = Field(..., pattern=r'^https://(www\.)?github\.com/[\w-]+/?$')
    profile_json: ProfileJson
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resume: Optional[bytes] = None
    resume_filename: Optional[str] = None

    @validator('created_at', 'updated_at')
    def datetime_to_iso(cls, v):
        return v.isoformat()

def validate_candidate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate candidate data against the schema.
    """
    try:
        validated_data = CandidateData(**data)
        return validated_data.dict()
    except Exception as e:
        raise ValueError(f"Invalid candidate data: {str(e)}") 