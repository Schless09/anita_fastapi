from typing import List
from .matching_types import Location, CandidateMetadata, JobMetadata, Dealbreakers

def check_location_match(job_cities: List[str], job_states: List[str], candidate_metadata: CandidateMetadata) -> bool:
    """Check if candidate's location preferences match the job location."""
    if not job_cities and not job_states:
        return True
        
    candidate_locations = candidate_metadata.get('preferred_locations', [])
    if not candidate_locations:
        return True
        
    # Convert to lowercase for comparison
    job_cities = [city.lower() for city in job_cities if city]
    job_states = [state.lower() for state in job_states if state]
    candidate_locations = [loc.lower() for loc in candidate_locations if loc]
    
    # Check if any candidate location matches job locations
    for location in candidate_locations:
        if location in job_cities or location in job_states:
            return True
    return False

def check_work_environment_match(job_environment: str, candidate_metadata: CandidateMetadata) -> bool:
    """Check if job work environment matches candidate's preference."""
    if not job_environment or job_environment.lower() == 'not specified':
        return True

    candidate_preference = candidate_metadata.get('preferred_work_environment')
    if not candidate_preference:
        return True

    return job_environment.lower() == candidate_preference.lower()

def extract_min_salary(salary_range: str) -> float:
    """Extract minimum salary from a salary range string."""
    try:
        # Handle various formats: "$120k-150k", "$120,000 - $150,000", etc.
        import re
        match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?', salary_range, re.IGNORECASE)
        if not match:
            return 0.0

        salary = float(match.group(1).replace(',', ''))
        if 'k' in salary_range.lower():
            salary *= 1000
        return salary
    except:
        return 0.0

def check_compensation_match(job_salary_range: str, candidate_metadata: CandidateMetadata) -> bool:
    """Check if job compensation matches candidate's minimum requirements."""
    if not job_salary_range:
        return True
        
    candidate_min_salary = candidate_metadata.get('minimum_salary')
    if not candidate_min_salary:
        return True
        
    try:
        # Extract minimum salary from range (assuming format like "$100k-$150k")
        job_min = int(job_salary_range.split('-')[0].replace('$', '').replace('k', '000'))
        candidate_min = int(candidate_min_salary.replace('$', '').replace('k', '000'))
        return job_min >= candidate_min
    except:
        return True

def check_work_authorization_match(job_work_auth: List[str], job_visa_sponsorship: str, candidate_metadata: CandidateMetadata) -> bool:
    """Check if candidate meets work authorization requirements."""
    if not job_work_auth and not job_visa_sponsorship:
        return True
        
    candidate_auth = candidate_metadata.get('work_authorization', [])
    if not candidate_auth:
        return True
        
    # If job offers visa sponsorship, always return True
    if job_visa_sponsorship and job_visa_sponsorship.lower() == 'yes':
        return True
        
    # Convert to lowercase for comparison
    job_auth = [auth.lower() for auth in job_work_auth] if isinstance(job_work_auth, list) else []
    candidate_auth = [auth.lower() for auth in candidate_auth] if isinstance(candidate_auth, list) else []
    
    # Check if candidate has any of the required authorizations
    return any(auth in job_auth for auth in candidate_auth)

def generate_match_reason(dealbreakers: Dealbreakers, job_metadata: JobMetadata) -> str:
    """Generate a human-readable explanation for the match."""
    reasons = []
    
    if all(dealbreakers.values()):
        reasons.append("All dealbreaker criteria met.")
    else:
        failed = [k for k, v in dealbreakers.items() if not v]
        reasons.append(f"Failed dealbreakers: {', '.join(failed)}")
    
    if job_metadata.get('tech_stack', {}).get('must_haves'):
        reasons.append(f"Required tech stack: {', '.join(job_metadata['tech_stack']['must_haves'])}")
    
    if job_metadata.get('experience_requirements', {}).get('minimum_years'):
        reasons.append(f"Required experience: {job_metadata['experience_requirements']['minimum_years']} years")
    
    return " ".join(reasons) 