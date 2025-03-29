from pydantic import BaseModel, EmailStr, HttpUrl, Field, AnyHttpUrl
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from fastapi import UploadFile, File

class CandidateBase(BaseModel):
    first_name: str = Field(..., description="Candidate's first name")
    last_name: str = Field(..., description="Candidate's last name")
    email: EmailStr = Field(..., description="Candidate's email address")
    phone: str = Field(..., description="Candidate's phone number")
    linkedin_url: Optional[AnyHttpUrl] = Field(None, description="Candidate's LinkedIn profile URL")

class CandidateCreate(CandidateBase):
    resume: Optional[UploadFile] = Field(None, description="Candidate's resume in PDF format (max 5MB)")

class CandidateUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    linkedin_url: Optional[AnyHttpUrl] = None
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