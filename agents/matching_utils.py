from typing import List
from .matching_types import Location, CandidateMetadata, JobMetadata, Dealbreakers

def check_location_match(job_cities: List[str], job_states: List[str], candidate_metadata: CandidateMetadata) -> bool:
    """Check if job location matches candidate's preferred locations."""
    # If job is remote, it's always a match
    if any(city.lower() in ['remote', 'any'] for city in job_cities) or \
       any(state.lower() in ['remote', 'any'] for state in job_states):
        return True

    # Extract candidate's preferred locations
    candidate_locations = candidate_metadata.get('preferred_locations', [])
    candidate_cities = [loc.city.lower() if loc.city else '' for loc in candidate_locations]
    candidate_states = [loc.state.lower() if loc.state else '' for loc in candidate_locations]

    # Check if any of the job locations match candidate's preferences
    return any(city.lower() in candidate_cities for city in job_cities) or \
           any(state.lower() in candidate_states for state in job_states)

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
    if not job_salary_range or job_salary_range.lower() == 'not specified':
        return True

    candidate_min_salary = candidate_metadata.get('minimum_salary')
    if not candidate_min_salary:
        return True

    job_min_salary = extract_min_salary(job_salary_range)
    if not job_min_salary:
        return True

    # Check if job's minimum salary is within $5k of candidate's minimum
    return abs(job_min_salary - candidate_min_salary) <= 5000

def check_work_authorization_match(required_auth: str, visa_sponsorship: str, candidate_metadata: CandidateMetadata) -> bool:
    """Check if candidate's work authorization matches job requirements."""
    if not required_auth or required_auth.lower() == 'not specified':
        return True

    candidate_auth = candidate_metadata.get('work_authorization')
    if not candidate_auth:
        return True

    # If candidate needs sponsorship, check if company offers it
    if 'need sponsorship' in candidate_auth.lower():
        return visa_sponsorship.lower() == 'yes'

    # If specific authorization is required, check if candidate has it
    return required_auth.lower() in candidate_auth.lower()

def generate_match_reason(dealbreakers: Dealbreakers, job_metadata: JobMetadata) -> str:
    """Generate a human-readable explanation for the match."""
    reasons = []
    
    if dealbreakers['location_match']:
        locations = job_metadata.get('role_details', {}).get('city', [])
        locations.extend(job_metadata.get('role_details', {}).get('state', []))
        if locations:
            reasons.append(f"Location match: {', '.join(locations)}")
            
    if dealbreakers['work_environment_match']:
        work_env = job_metadata.get('company_information', {}).get('company_culture', {}).get('work_environment')
        if work_env:
            reasons.append(f"Work environment match: {work_env}")
            
    if dealbreakers['compensation_match']:
        salary = job_metadata.get('role_details', {}).get('salary_range')
        if salary:
            reasons.append(f"Compensation match: {salary}")
            
    if dealbreakers['work_authorization_match']:
        auth = job_metadata.get('role_details', {}).get('work_authorization')
        if auth:
            reasons.append(f"Work authorization match: {auth}")
            
    return " | ".join(reasons) if reasons else "General match based on skills and experience" 