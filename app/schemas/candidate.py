from pydantic import BaseModel, EmailStr, HttpUrl, Field, AnyHttpUrl, validator
from typing import List, Optional, Dict, Any, Annotated, Union
from datetime import datetime
from fastapi import UploadFile, File
import base64
import uuid
from enum import Enum

# --- Define Enums based on allowed values ---

class WorkEnvironmentEnum(str, Enum):
    ONSITE = 'Onsite'
    HYBRID = 'Hybrid'
    REMOTE = 'Fully Remote'

class WorkAuthorizationEnum(str, Enum):
    AUTH_NO_SPONSOR = 'Authorized_NoSponsorship'
    REQ_NOW = 'RequireSponsorshipNow'
    REQ_FUTURE = 'RequireSponsorshipFuture'
    NOT_SURE = 'NotSure'

class VisaTypeEnum(str, Enum):
    H1B = 'H-1B'
    H1B_TRANSFER = 'H-1B (Transfer)'
    F1_OPT = 'F-1 OPT'
    F1_STEM_OPT = 'F-1 STEM OPT'
    CPT = 'CPT'
    L1 = 'L-1'
    TN = 'TN'
    O1 = 'O-1'
    J1 = 'J-1'
    E3 = 'E-3'
    H4_NO_EAD = 'H-4 (without EAD)'
    OTHER = 'Other'

class EmploymentTypeEnum(str, Enum):
    FULL_TIME = 'Full-time'
    CONTRACT = 'Contract'
    CONTRACT_TO_HIRE = 'Contract-To-Hire'

class AvailabilityEnum(str, Enum):
    ASAP = 'ASAP'
    WITHIN_3_MONTHS = 'Within3Months'
    MONTHS_4_TO_6 = '4to6Months'
    MONTHS_6_PLUS = '6PlusMonths'

# --- Existing Schemas ---

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

class CandidateCreate(BaseModel):
    id: str = Field(..., description="Unique identifier for the candidate")
    name: str = Field(..., description="Candidate's full name")
    email: EmailStr = Field(..., description="Candidate's email address")
    phone_number: str = Field(..., description="Candidate's phone number in E.164 format")
    linkedin: Optional[HttpUrl] = Field(None, description="Candidate's LinkedIn profile URL")
    resume_content: bytes = Field(..., description="Binary content of the resume PDF")
    resume_filename: str = Field(..., description="Original filename of the resume")
    work_environment: Optional[List[WorkEnvironmentEnum]] = Field(None, description="Array of preferred work environments")
    desired_locations: Optional[List[str]] = Field(None, description="Array of preferred main geographic locations")
    preferred_sub_locations: Optional[List[str]] = Field(None, description="Array of preferred sub-locations")
    work_authorization: Optional[WorkAuthorizationEnum] = Field(None, description="Candidate's US work authorization status")
    visa_type: Optional[VisaTypeEnum] = Field(None, description="Candidate's specific visa type if sponsorship is required")
    employment_types: Optional[List[EmploymentTypeEnum]] = Field(None, description="Array of desired employment types")
    availability: Optional[AvailabilityEnum] = Field(None, description="Candidate's target start timeline")
    dream_role_description: Optional[str] = Field(None, description="Candidate's description of their ideal role")

    @validator('phone_number')
    def validate_phone(cls, v):
        # Remove any non-digit characters
        cleaned = ''.join(filter(str.isdigit, v))
        # Add +1 prefix if not present
        if not cleaned.startswith('1'):
            cleaned = '1' + cleaned
        return f"+{cleaned}"

    class Config:
        json_encoders = {
            bytes: lambda v: base64.b64encode(v).decode('utf-8')
        }

class CandidateUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None
    linkedin: Optional[HttpUrl] = None
    resume_url: Optional[str] = None
    work_environment: Optional[List[WorkEnvironmentEnum]] = None
    desired_locations: Optional[List[str]] = None
    preferred_sub_locations: Optional[List[str]] = None
    work_authorization: Optional[WorkAuthorizationEnum] = None
    visa_type: Optional[VisaTypeEnum] = None
    employment_types: Optional[List[EmploymentTypeEnum]] = None
    availability: Optional[AvailabilityEnum] = None
    dream_role_description: Optional[str] = None

class CandidateResponse(CandidateBase):
    id: str
    resume_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    status: str = "pending"
    profile_json: Optional[Dict[str, Any]] = None
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    work_environment: Optional[List[WorkEnvironmentEnum]] = None
    desired_locations: Optional[List[str]] = None
    preferred_sub_locations: Optional[List[str]] = None
    work_authorization: Optional[WorkAuthorizationEnum] = None
    visa_type: Optional[VisaTypeEnum] = None
    employment_types: Optional[List[EmploymentTypeEnum]] = None
    availability: Optional[AvailabilityEnum] = None
    dream_role_description: Optional[str] = None

    class Config:
        from_attributes = True

# Add the missing schema
class CandidateStatusUpdate(BaseModel):
    candidate_id: uuid.UUID # Assuming UUID based on context
    status: str 