from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated

class Match(BaseModel):
    candidate_id: Optional[str] = None
    job_id: Optional[str] = None
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    metadata: Optional[Dict[str, Any]] = None

class MatchingResponse(BaseModel):
    status: str
    matches: List[Match]
    metadata: Optional[Dict[str, Any]] = None 