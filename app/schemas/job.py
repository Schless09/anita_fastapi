from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime

class SalaryRange(BaseModel):
    min: int
    max: int

class JobBase(BaseModel):
    title: str
    company: str
    description: str
    requirements: Annotated[List[str], Field(min_items=1)]
    location: str
    salary_range: Optional[SalaryRange] = None
    employment_type: str

class JobCreate(JobBase):
    pass

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