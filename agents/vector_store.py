import os
from pinecone import Pinecone
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI

def check_location_match(job_cities: List[str], job_states: List[str], candidate_metadata: Dict[str, Any]) -> bool:
    """Check if candidate's location preferences match the job location."""
    if not job_cities or not job_states:
        return True  # If job location is not specified, consider it a match
    
    candidate_locations = candidate_metadata.get('preferred_locations', [])
    if not candidate_locations:
        return True  # If candidate has no location preferences, consider it a match
    
    for location in candidate_locations:
        if (location.get('city', '').lower() in [city.lower() for city in job_cities] or
            location.get('state', '').lower() in [state.lower() for state in job_states]):
            return True
    return False

def check_work_environment_match(job_environment: str, candidate_metadata: Dict[str, Any]) -> bool:
    """Check if candidate's preferred work environment matches the job."""
    if not job_environment:
        return True
    
    candidate_preference = candidate_metadata.get('preferred_work_environment', '').lower()
    if not candidate_preference:
        return True
    
    return job_environment.lower() in candidate_preference

def check_compensation_match(job_salary_range: str, candidate_metadata: Dict[str, Any]) -> bool:
    """Check if job's salary range meets candidate's minimum requirements."""
    if not job_salary_range:
        return True
    
    candidate_min_salary = candidate_metadata.get('minimum_salary')
    if not candidate_min_salary:
        return True
    
    # Extract minimum salary from job range (assuming format like "100000-150000")
    try:
        job_min_salary = float(job_salary_range.split('-')[0].strip())
        return job_min_salary >= candidate_min_salary
    except (ValueError, IndexError):
        return True

def check_work_authorization_match(required_authorization: str, visa_sponsorship: str, candidate_metadata: Dict[str, Any]) -> bool:
    """Check if candidate's work authorization matches job requirements."""
    if not required_authorization:
        return True
    
    candidate_authorization = candidate_metadata.get('work_authorization', '').lower()
    if not candidate_authorization:
        return True
    
    if 'citizen' in candidate_authorization or 'permanent resident' in candidate_authorization:
        return True
    
    return visa_sponsorship.lower() == 'yes'

def generate_match_reason(dealbreakers: Dict[str, bool], job_metadata: Dict[str, Any]) -> str:
    """Generate a human-readable explanation for why this is a good match."""
    reasons = []
    
    if dealbreakers.get('location_match'):
        location = job_metadata.get('role_details', {}).get('city', [''])[0]
        reasons.append(f"Location match with {location}")
    
    if dealbreakers.get('work_environment_match'):
        env = job_metadata.get('company_information', {}).get('company_culture', {}).get('work_environment', '')
        reasons.append(f"Preferred work environment: {env}")
    
    if dealbreakers.get('compensation_match'):
        salary = job_metadata.get('role_details', {}).get('salary_range', '')
        reasons.append(f"Salary requirements met: {salary}")
    
    if dealbreakers.get('work_authorization_match'):
        reasons.append("Work authorization requirements met")
    
    return "; ".join(reasons) if reasons else "General match based on experience and skills"

class VectorStore:
    def __init__(self):
        """Initialize Pinecone with environment variables."""
        print("Initializing Pinecone...")
        
        # Initialize Pinecone with new SDK v3
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        print("Pinecone initialized successfully")
        print("Attempting to connect to indexes: anita-candidates, job-details")
        
        # Get list of existing indexes
        existing_indexes = self.pc.list_indexes()
        print("Existing indexes:", existing_indexes)
        
        # Connect to indexes
        self.candidates_index = self.pc.Index("anita-candidates")
        self.jobs_index = self.pc.Index("job-details")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's API."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            raise

    def store_candidate(self, candidate_id: str, candidate_data: Dict[str, Any]) -> Dict[str, str]:
        """Store candidate data in vector database."""
        try:
            # Create text representation for embedding
            text_for_embedding = f"{candidate_data.get('name', '')} {candidate_data.get('resume_text', '')}"
            
            # Get embedding
            vector = self.get_embedding(text_for_embedding)
            
            # Store in Pinecone
            self.candidates_index.upsert(
                vectors=[(
                    candidate_id,
                    vector,
                    {
                        "name": candidate_data.get("name"),
                        "email": candidate_data.get("email"),
                        "phone_number": candidate_data.get("phone_number"),
                        "linkedin": candidate_data.get("linkedin"),
                        "resume_text": candidate_data.get("resume_text"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )]
            )
            
            return {
                "status": "success",
                "message": f"Candidate {candidate_id} stored successfully"
            }
            
        except Exception as e:
            print(f"Error storing candidate: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def find_similar_jobs(self, candidate_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find jobs similar to a candidate's profile."""
        try:
            # Get candidate vector
            candidate_data = self.candidates_index.fetch([candidate_id])
            if not candidate_data.vectors:
                return {
                    "status": "error",
                    "message": f"Candidate {candidate_id} not found"
                }
            
            # Query jobs index
            results = self.jobs_index.query(
                vector=candidate_data.vectors[candidate_id].values,
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    "job_id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return {
                "status": "success",
                "matches": matches
            }
            
        except Exception as e:
            print(f"Error finding similar jobs: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def find_similar_candidates(self, job_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find candidates that match a job posting with enhanced matching"""
        try:
            # Get job data
            job_response = self.jobs_index.fetch(ids=[job_id])
            if not job_response.vectors:
                return {"status": "error", "message": "Job not found"}
            
            job_vector = job_response.vectors[job_id].values
            job_metadata = job_response.vectors[job_id].metadata
            
            # Query candidates using vector similarity
            query_response = self.candidates_index.query(
                vector=job_vector,
                top_k=20,  # Get more matches initially for filtering
                include_metadata=True
            )
            
            matches = []
            
            for match in query_response.matches:
                if not match.metadata:
                    continue
                
                candidate_metadata = match.metadata
                role_details = job_metadata.get('role_details', {})
                
                # Check dealbreakers
                dealbreakers = {
                    'location_match': check_location_match(
                        role_details.get('city', []),
                        role_details.get('state', []),
                        candidate_metadata
                    ),
                    'work_environment_match': check_work_environment_match(
                        job_metadata.get('company_information', {}).get('company_culture', {}).get('work_environment', ''),
                        candidate_metadata
                    ),
                    'compensation_match': check_compensation_match(
                        role_details.get('salary_range', ''),
                        candidate_metadata
                    ),
                    'work_authorization_match': check_work_authorization_match(
                        role_details.get('work_authorization', ''),
                        role_details.get('visa_sponsorship', ''),
                        candidate_metadata
                    )
                }
                
                # Only include matches that pass all dealbreakers
                if all(dealbreakers.values()):
                    matches.append({
                        'candidate_id': match.id,
                        'score': float(match.score),
                        'metadata': candidate_metadata,
                        'dealbreakers': dealbreakers,
                        'match_reason': generate_match_reason(dealbreakers, job_metadata)
                    })
            
            # Sort by score and return top_k
            matches.sort(key=lambda x: x['score'], reverse=True)
            return {
                "status": "success",
                "matches": matches[:top_k]
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _create_candidate_text(self, candidate_data: Dict[str, Any]) -> str:
        """Create a text representation of candidate data for embedding"""
        parts = []
        
        # Add basic info
        if 'name' in candidate_data:
            parts.append(f"Name: {candidate_data['name']}")
        if 'email' in candidate_data:
            parts.append(f"Email: {candidate_data['email']}")
        if 'phone_number' in candidate_data:
            parts.append(f"Phone: {candidate_data['phone_number']}")
        if 'linkedin' in candidate_data and candidate_data['linkedin']:
            parts.append(f"LinkedIn: {candidate_data['linkedin']}")
        if 'resume' in candidate_data and candidate_data['resume']:
            parts.append(f"Resume: {candidate_data['resume']}")
            
        # Join all parts with newlines
        return '\n'.join(parts) 