import os
from pinecone import Pinecone
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI

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