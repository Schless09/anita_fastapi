import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from .matching_types import JobMatch, JobMetadata, CandidateMetadata, Dealbreakers
from .matching_utils import (
    check_location_match,
    check_work_environment_match,
    check_compensation_match,
    check_work_authorization_match,
    generate_match_reason
)
import openai
from fastapi import HTTPException
from datetime import datetime

class VectorStore:
    def __init__(self):
        """Initialize connection to Pinecone and set up indexes."""
        print("Initializing Pinecone with environment: us-east-1")
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        print("Pinecone initialized successfully")
        
        print("Attempting to connect to indexes: anita-candidates, job-details")
        try:
            existing_indexes = self.pc.list_indexes()
            print(f"Existing indexes: {existing_indexes}")
            
            print("Connecting to indexes...")
            self.candidates_index = self.pc.Index("anita-candidates")
            self.jobs_index = self.pc.Index("job-details")
            print("Successfully connected to indexes")
            
        except Exception as e:
            print(f"Error setting up Pinecone indexes: {str(e)}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Convert text to vector embedding using OpenAI's text-embedding-3-small model"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def store_candidate(self, candidate_id: str, candidate_data: Dict[str, Any]) -> Dict[str, str]:
        """Store candidate data in vector database with enhanced metadata"""
        try:
            # Create a combined text representation of the candidate
            text_repr = self._create_candidate_text(candidate_data)
            
            # Truncate text if it's too long (OpenAI's limit is 8192 tokens)
            max_chars = 6000  # Conservative estimate to stay under token limit
            if len(text_repr) > max_chars:
                text_repr = text_repr[:max_chars]
            
            # Get vector embedding
            print(f"Getting embedding for candidate {candidate_id}")
            vector = self.get_embedding(text_repr)
            print(f"Successfully got embedding for candidate {candidate_id}")
            
            # Generate a default ID if none provided
            if not candidate_id or candidate_id == "default_id":
                candidate_id = f"candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare metadata in the correct format
            name_parts = candidate_data.get('name', '').split() if candidate_data.get('name') else []
            first_name = name_parts[0] if name_parts else ''
            last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
            
            metadata = {
                'first_name': first_name,
                'last_name': last_name,
                'email': candidate_data.get('email', ''),
                'phone': candidate_data.get('phone_number', ''),
                'linkedin': candidate_data.get('linkedin', ''),
                'resume_text': candidate_data.get('resume_text', '')[:1000] if candidate_data.get('resume_text') else '',  # Truncate resume text
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Ensure all metadata values are valid types
            for key, value in metadata.items():
                if value is None:
                    metadata[key] = ''
                elif isinstance(value, (list, tuple)):
                    metadata[key] = [str(v) for v in value]
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            
            print(f"Storing candidate {candidate_id} in Pinecone")
            # Store in Pinecone
            self.candidates_index.upsert(
                vectors=[{
                    'id': candidate_id,
                    'values': vector,
                    'metadata': metadata
                }]
            )
            print(f"Successfully stored candidate {candidate_id} in Pinecone")
            
            return {"status": "success", "message": f"Stored candidate {candidate_id} in vector database"}
        except Exception as e:
            print(f"Error storing candidate {candidate_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def find_similar_jobs(self, candidate_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find similar job postings for a candidate with enhanced matching"""
        try:
            # Get candidate data
            candidate_response = self.candidates_index.fetch(ids=[candidate_id])
            if not candidate_response.vectors:
                return {"status": "error", "message": "Candidate not found"}
            
            candidate_vector = candidate_response.vectors[candidate_id].values
            candidate_metadata = candidate_response.vectors[candidate_id].metadata
            
            # Query jobs using vector similarity
            query_response = self.jobs_index.query(
                vector=candidate_vector,
                top_k=20,  # Get more matches initially for filtering
                include_metadata=True
            )
            
            matches: List[JobMatch] = []
            
            for match in query_response.matches:
                if not match.metadata:
                    continue
                
                job_metadata = match.metadata
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
                        'job_id': match.id,
                        'score': float(match.score),
                        'metadata': job_metadata,
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