from pydantic import BaseModel, EmailStr, HttpUrl, Field, AnyHttpUrl, validator
from typing import List, Optional, Dict, Any, Annotated, Union
from datetime import datetime
from fastapi import UploadFile, File

class CandidateBase(BaseModel):
    name: str = Field(..., description="Candidate's full name")
    email: EmailStr = Field(..., description="Candidate's email address")
    phone_number: str = Field(..., description="Candidate's phone number in E.164 format")
    linkedin: Optional[HttpUrl] = Field(None, description="Candidate's LinkedIn profile URL")

    @validator('phone_number')
    def validate_phone(cls, v):
        # Remove any non-digit characters
        cleaned = ''.join(filter(str.isdigit, v))
        # Add +1 prefix if not present
        if not cleaned.startswith('1'):
            cleaned = '1' + cleaned
        return f"+{cleaned}"

class CandidateCreate(CandidateBase):
    resume_content: Optional[bytes] = Field(None, description="Binary content of the candidate's resume")
    resume_filename: Optional[str] = Field(None, description="Filename of the uploaded resume")
    
    class Config:
        arbitrary_types_allowed = True

class CandidateUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None
    linkedin: Optional[HttpUrl] = None
    resume_url: Optional[HttpUrl] = None

class CandidateResponse(CandidateBase):
    id: str
    resume_url: Optional[HttpUrl] = None
    created_at: datetime
    updated_at: datetime
    status: str = "pending"
    profile_json: Optional[Dict[str, Any]] = None
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True 