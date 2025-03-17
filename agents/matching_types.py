from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class WorkEnvironment(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    NOT_SPECIFIED = "not_specified"

@dataclass
class Location:
    city: Optional[str] = None
    state: Optional[str] = None

class JobMetadata(TypedDict, total=False):
    role_details: Dict[str, any]  # Includes city, state, salary_range, work_authorization
    company_information: Dict[str, any]  # Includes company_culture, work_environment

class CandidateMetadata(TypedDict, total=False):
    first_name: str
    last_name: str
    email: str
    phone: str
    linkedin: str
    resume_text: str
    timestamp: str
    preferred_locations: List[Location]
    preferred_work_environment: str
    minimum_salary: float
    work_authorization: str

class Dealbreakers(TypedDict):
    location_match: bool
    work_environment_match: bool
    compensation_match: bool
    work_authorization_match: bool

class JobMatch(TypedDict):
    job_id: str
    score: float
    metadata: JobMetadata
    dealbreakers: Dealbreakers
    match_reason: str 