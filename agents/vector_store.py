import os
from pinecone import Pinecone
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
import time
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from openai import OpenAI
import json
import logging
from .matching_utils import (
    check_location_match,
    check_work_environment_match,
    check_compensation_match,
    check_work_authorization_match,
    generate_match_reason
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class VectorStore:
    def __init__(self, init_openai: bool = False, existing_indexes: Optional[Dict[str, Any]] = None):
        """Initialize VectorStore with optional existing Pinecone instances."""
        if existing_indexes:
            # Use existing Pinecone instances
            self.candidates_index = existing_indexes.get('candidates_index')
            self.jobs_index = existing_indexes.get('jobs_index')
            self.call_status_index = existing_indexes.get('call_status_index')
            
            if not all([self.candidates_index, self.jobs_index, self.call_status_index]):
                raise ValueError("Missing required Pinecone indexes")
        else:
            # Initialize Pinecone if no existing instances provided
            print("Initializing Pinecone...")
            
            # Load environment variables
            load_dotenv()
            
            # Initialize Pinecone
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
                
            self.pc = Pinecone(api_key=pinecone_api_key)
            
            # Get index names from environment variables
            candidates_index_name = os.getenv('PINECONE_CANDIDATES_INDEX')
            jobs_index_name = os.getenv('PINECONE_JOBS_INDEX')
            call_status_index_name = os.getenv('PINECONE_CALL_STATUS_INDEX', 'call-statuses')
            
            # Validate environment variables
            if not candidates_index_name or not jobs_index_name:
                raise ValueError("Missing required environment variables: PINECONE_CANDIDATES_INDEX, PINECONE_JOBS_INDEX")
            
            print(f"Pinecone initialized successfully")
            print(f"Attempting to connect to indexes: {candidates_index_name}, {jobs_index_name}")
            
            # Get list of existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            print("Existing indexes:", existing_indexes)
            
            # Validate indexes exist
            if candidates_index_name not in existing_indexes:
                raise ValueError(f"Candidates index '{candidates_index_name}' does not exist")
            if jobs_index_name not in existing_indexes:
                raise ValueError(f"Jobs index '{jobs_index_name}' does not exist")
            if call_status_index_name not in existing_indexes:
                raise ValueError(f"Call status index '{call_status_index_name}' does not exist")
            
            # Connect to indexes
            self.candidates_index = self.pc.Index(candidates_index_name)
            self.jobs_index = self.pc.Index(jobs_index_name)
            self.call_status_index = self.pc.Index(call_status_index_name)
        
        if init_openai:
            # Check for OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. Required for storing candidates.")
            
            # Initialize OpenAI client with new API format
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            
            # Initialize tokenizer for text-embedding-3-small
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Set chunk size (in tokens)
            self.chunk_size = 500  # Conservative chunk size to stay well under limits
            
            print("OpenAI client initialized successfully")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_size += 1
            
            if current_size >= self.chunk_size:
                chunks.append(self.tokenizer.decode(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(self.tokenizer.decode(current_chunk))
            
        return chunks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's API with retry logic."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            if "rate_limit_exceeded" in str(e):
                print("Rate limit exceeded, waiting before retry...")
                time.sleep(2)  # Add delay before retry
            raise

    def store_candidate(self, candidate_id: str, candidate_data: Dict[str, Any]) -> Dict[str, str]:
        """Store candidate data in vector database with chunking."""
        try:
            print(f"\n=== Storing candidate {candidate_id} ===")
            # Validate required fields
            required_fields = ['name', 'email', 'phone_number']
            missing_fields = [field for field in required_fields if not candidate_data.get(field)]
            if missing_fields:
                print(f"Missing required fields: {', '.join(missing_fields)}")
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Create text representation for embedding
            name = candidate_data.get('name', '')
            resume_text = candidate_data.get('resume_text', '')
            print(f"Processing candidate: {name}")
            print(f"Resume text length: {len(resume_text)}")
            
            # Ensure candidate_id is in the metadata
            metadata = {
                "name": name,
                "email": candidate_data.get("email"),
                "phone_number": candidate_data.get("phone_number"),
                "candidate_id": candidate_id,  # Add this for easier querying
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Only add linkedin if it has a value
            if candidate_data.get("linkedin"):
                metadata["linkedin"] = candidate_data.get("linkedin")
            
            # If no resume text, create a basic profile vector
            if not resume_text:
                print("No resume text found, creating basic profile vector")
                profile_text = f"{name} {candidate_data.get('email', '')} {candidate_data.get('phone_number', '')}"
                print(f"Creating embedding for profile text: {profile_text}")
                vector = self.get_embedding(profile_text)
                print("Successfully created embedding")
                
                # Store single vector for basic profile
                print("Storing basic profile in Pinecone...")
                metadata.update({
                    "resume_text": "",
                    "chunk_index": 0,
                    "total_chunks": 1
                })
                
                self.candidates_index.upsert(vectors=[(
                    candidate_id,  # Use the original ID without chunk suffix
                    vector,
                    metadata
                )])
                print(f"Successfully stored basic profile with ID: {candidate_id}")
                
                return {
                    "status": "success",
                    "message": f"Candidate {candidate_id} stored successfully (basic profile)"
                }
            
            # Split resume text into chunks
            print("Splitting resume text into chunks...")
            chunks = self.chunk_text(resume_text)
            print(f"Created {len(chunks)} chunks")
            
            # Create vectors for each chunk
            vectors = []
            for i, chunk in enumerate(chunks):
                print(f"\nProcessing chunk {i+1}/{len(chunks)}")
                # Combine name with chunk for better context
                chunk_text = f"{name} {chunk}"
                print(f"Creating embedding for chunk (length: {len(chunk_text)})")
                vector = self.get_embedding(chunk_text)
                print("Successfully created embedding")
                
                # Create unique ID for each chunk
                chunk_id = f"{candidate_id}_chunk_{i}" if len(chunks) > 1 else candidate_id
                
                # Update metadata for this chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "resume_text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                vectors.append((
                    chunk_id,
                    vector,
                    chunk_metadata
                ))
                print(f"Added chunk {i+1} to vectors list with ID: {chunk_id}")
                
                # Add small delay between chunks to avoid rate limits
                if i < len(chunks) - 1:
                    time.sleep(0.5)
            
            # Store all chunks in Pinecone
            print(f"\nStoring {len(vectors)} vectors in Pinecone...")
            self.candidates_index.upsert(vectors=vectors)
            print("Successfully stored all chunks in Pinecone")
            
            # Verify storage
            print("\nVerifying storage...")
            verification = self.candidates_index.query(
                vector=[0] * 1536,
                filter={"candidate_id": candidate_id},
                top_k=1,
                include_metadata=True
            )
            if verification.matches:
                print(f"Successfully verified storage. Found candidate with ID: {verification.matches[0].id}")
            else:
                print("Warning: Could not verify storage immediately. This might be due to indexing delay.")
            
            return {
                "status": "success",
                "message": f"Candidate {candidate_id} stored successfully in {len(chunks)} chunks"
            }
            
        except Exception as e:
            print(f"Error storing candidate: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            print("=== Store candidate operation complete ===\n")

    def get_candidate_chunks(self, candidate_id: str) -> Dict[str, Any]:
        """Retrieve all chunks for a candidate and reconstruct the full profile."""
        try:
            print(f"\n=== Retrieving chunks for candidate {candidate_id} ===")
            
            # Query all chunks for this candidate
            chunks = self.candidates_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata query
                filter={"candidate_id": candidate_id},
                include_metadata=True,
                top_k=10  # Adjust based on your max chunks per candidate
            )
            
            if not chunks.matches:
                return {
                    "status": "error",
                    "message": f"No chunks found for candidate {candidate_id}"
                }
            
            # Sort chunks by index
            sorted_chunks = sorted(
                chunks.matches,
                key=lambda x: x.metadata.get("chunk_index", 0)
            )
            
            # Reconstruct the full profile
            full_resume = " ".join(chunk.metadata.get("resume_text", "") for chunk in sorted_chunks)
            
            # Get the first chunk's metadata (contains the main profile info)
            main_metadata = sorted_chunks[0].metadata
            
            return {
                "status": "success",
                "candidate_id": candidate_id,
                "total_chunks": len(sorted_chunks),
                "profile": {
                    "name": main_metadata.get("name"),
                    "email": main_metadata.get("email"),
                    "phone_number": main_metadata.get("phone_number"),
                    "linkedin": main_metadata.get("linkedin"),
                    "timestamp": main_metadata.get("timestamp")
                },
                "resume_text": full_resume
            }
            
        except Exception as e:
            print(f"Error retrieving candidate chunks: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def find_similar_jobs(self, candidate_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find jobs similar to a candidate's profile."""
        try:
            print(f"\n=== Finding jobs for candidate {candidate_id} ===")
            
            # Get all chunks for the candidate
            candidate_data = self.get_candidate_chunks(candidate_id)
            if candidate_data["status"] == "error":
                return candidate_data
                
            # Create a single vector from the full resume
            full_resume = candidate_data["resume_text"]
            vector = self.get_embedding(full_resume)
            
            # Search for similar jobs
            similar_jobs = self.jobs_index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )
            
            return {
                "status": "success",
                "candidate_id": candidate_id,
                "matches": [
                    {
                        "job_id": match.id,
                        "score": match.score,
                        "metadata": match.metadata
                    }
                    for match in similar_jobs.matches
                ]
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
            # We'll get more matches initially to account for chunked vectors
            query_response = self.candidates_index.query(
                vector=job_vector,
                top_k=20,  # Get more matches initially for filtering
                include_metadata=True
            )
            
            # Group matches by candidate ID (removing chunk suffix)
            candidate_matches = {}
            for match in query_response.matches:
                if not match.metadata:
                    continue
                
                # Get base candidate ID (remove chunk suffix)
                base_candidate_id = match.id.split('_chunk_')[0]
                
                if base_candidate_id not in candidate_matches:
                    candidate_matches[base_candidate_id] = {
                        'score': match.score,
                        'metadata': match.metadata,
                        'chunks': []
                    }
                else:
                    # Update score if this chunk has a better match
                    if match.score > candidate_matches[base_candidate_id]['score']:
                        candidate_matches[base_candidate_id]['score'] = match.score
                    candidate_matches[base_candidate_id]['chunks'].append(match.metadata)
            
            matches = []
            for candidate_id, match_data in candidate_matches.items():
                candidate_metadata = match_data['metadata']
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
                        'candidate_id': candidate_id,
                        'score': float(match_data['score']),
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

    def update_candidate_profile(self, candidate_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a candidate's profile in the vector store with additional data."""
        try:
            # Query for existing vectors with this candidate_id
            query_response = self.candidates_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata-only query
                filter={"candidate_id": candidate_id},
                top_k=1,
                include_metadata=True
            )
            
            if not query_response.matches:
                print(f"No existing vectors found for candidate {candidate_id}")
                return {
                    "status": "error",
                    "message": "Candidate not found in vector store"
                }
            
            # Get the first match (should be the most recent)
            match = query_response.matches[0]
            
            # Update the metadata with new data
            current_metadata = match.metadata
            current_metadata.update(update_data)
            
            # Update the vector with new metadata
            self.candidates_index.update(
                id=match.id,
                metadata=current_metadata
            )
            
            return {
                "status": "success",
                "message": "Candidate profile updated successfully"
            }
            
        except Exception as e:
            print(f"Error updating candidate profile: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_candidate_profile(self, candidate_id: str) -> Dict[str, Any]:
        """Get a candidate's complete profile including all chunks and metadata."""
        try:
            print(f"\n=== Retrieving profile for candidate {candidate_id} ===")
            
            # Get all chunks for the candidate
            chunks_data = self.get_candidate_chunks(candidate_id)
            if chunks_data["status"] == "error":
                return chunks_data
            
            # Get the full resume text
            full_resume = chunks_data["resume_text"]
            
            # Process the resume with OpenAI to extract structured information
            try:
                processed_data = self.process_resume_with_openai(full_resume)
            except Exception as e:
                print(f"Warning: Failed to process resume with OpenAI: {str(e)}")
                processed_data = None
            
            # Combine all data
            profile = {
                "status": "success",
                "candidate_id": candidate_id,
                "basic_info": chunks_data["profile"],
                "resume_text": full_resume,
                "processed_data": processed_data,
                "total_chunks": chunks_data["total_chunks"]
            }
            
            return profile
            
        except Exception as e:
            print(f"Error retrieving candidate profile: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def process_resume_with_openai(self, resume_text: str) -> Dict[str, Any]:
        """Process resume text with OpenAI to extract structured information."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": """You are an expert resume analyzer. Extract the following information in a structured format:
                    - skills: List of technical and soft skills
                    - experience: List of work experiences with company, role, and duration
                    - education: List of educational qualifications
                    - achievements: List of key achievements
                    - summary: A brief professional summary

                    Format the response as a valid JSON object with these exact keys."""},
                    {"role": "user", "content": f"Please analyze this resume and extract the key information:\n\n{resume_text}"}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={ "type": "json_object" }
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error processing resume with OpenAI: {str(e)}")
            raise

    def store_job(self, job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a job posting in the vector database."""
        try:
            # Create text representation of the job for embedding
            job_text = f"""
            Job Title: {job_data.get('job_title', 'n/a')}
            Company: {job_data.get('company_name', 'n/a')}
            Location: {', '.join(job_data.get('city', ['n/a']))}, {', '.join(job_data.get('state', ['n/a']))}
            Description: {job_data.get('job_description', 'n/a')}
            Requirements: {', '.join(job_data.get('tech_stack_must_haves', ['n/a']))}
            Nice to Have: {', '.join(job_data.get('tech_stack_nice_to_haves', ['n/a']))}
            Experience: {job_data.get('minimum_years_of_experience', 'n/a')}
            Salary: {job_data.get('salary_range', 'n/a')}
            Equity: {job_data.get('equity_range', 'n/a')}
            Work Arrangement: {', '.join(job_data.get('work_arrangement', ['n/a']))}
            Role Category: {', '.join(job_data.get('role_category', ['n/a']))}
            """
            
            # Generate embedding for the job text
            embedding = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=job_text
            ).data[0].embedding
            
            # Create metadata dictionary with all fields
            metadata = {
                "job_id": job_id,
                "job_title": job_data.get('job_title', 'n/a'),
                "company_name": job_data.get('company_name', 'n/a'),
                "job_url": job_data.get('job_url', 'n/a'),
                "company_url": job_data.get('company_url', 'n/a'),
                "company_stage": job_data.get('company_stage', ['n/a']),
                "most_recent_funding_round_amount": job_data.get('most_recent_funding_round_amount', 'n/a'),
                "total_funding_amount": job_data.get('total_funding_amount', 'n/a'),
                "investors": job_data.get('investors', ['n/a']),
                "team_size": job_data.get('team_size', 'n/a'),
                "founding_year": job_data.get('founding_year', 'n/a'),
                "company_mission": job_data.get('company_mission', 'n/a'),
                "target_market": job_data.get('target_market', ['n/a']),
                "industry_vertical": job_data.get('industry_vertical', 'n/a'),
                "company_vision": job_data.get('company_vision', 'n/a'),
                "company_growth_story": job_data.get('company_growth_story', 'n/a'),
                "work_environment": job_data.get('company_culture', {}).get('work_environment', 'n/a'),
                "decision_making": job_data.get('company_culture', {}).get('decision_making', 'n/a'),
                "collaboration_style": job_data.get('company_culture', {}).get('collaboration_style', 'n/a'),
                "risk_tolerance": job_data.get('company_culture', {}).get('risk_tolerance', 'n/a'),
                "company_values": job_data.get('company_culture', {}).get('values', 'n/a'),
                "scaling_plans": job_data.get('scaling_plans', 'n/a'),
                "mission_and_impact": job_data.get('mission_and_impact', 'n/a'),
                "tech_innovation": job_data.get('tech_innovation', 'n/a'),
                "positions_available": job_data.get('positions_available', 'n/a'),
                "hiring_urgency": job_data.get('hiring_urgency', ['n/a']),
                "seniority_level": job_data.get('seniority_level', ['n/a']),
                "work_arrangement": job_data.get('work_arrangement', ['n/a']),
                "city": job_data.get('city', ['n/a']),
                "state": job_data.get('state', ['n/a']),
                "visa_sponsorship": job_data.get('visa_sponsorship', 'n/a'),
                "work_authorization": job_data.get('work_authorization', 'n/a'),
                "salary_range": job_data.get('salary_range', 'n/a'),
                "equity_range": job_data.get('equity_range', 'n/a'),
                "reporting_structure": job_data.get('reporting_structure', 'n/a'),
                "team_composition": job_data.get('team_composition', 'n/a'),
                "role_status": job_data.get('role_status', 'n/a'),
                "role_category": job_data.get('role_category', ['n/a']),
                "tech_stack_must_haves": job_data.get('tech_stack_must_haves', ['n/a']),
                "tech_stack_nice_to_haves": job_data.get('tech_stack_nice_to_haves', ['n/a']),
                "tech_stack_tags": job_data.get('tech_stack_tags', ['n/a']),
                "tech_breadth_requirement": job_data.get('tech_breadth_requirement', ['n/a']),
                "minimum_years_of_experience": job_data.get('minimum_years_of_experience', 'n/a'),
                "domain_expertise": job_data.get('domain_expertise', ['n/a']),
                "ai_ml_experience": job_data.get('ai_ml_experience', 'n/a'),
                "infrastructure_experience": job_data.get('infrastructure_experience', ['n/a']),
                "system_design_level": job_data.get('system_design_level', 'n/a'),
                "coding_proficiency_required": job_data.get('coding_proficiency_required', ['n/a']),
                "coding_languages_versions": job_data.get('coding_languages_versions', ['n/a']),
                "version_control_experience": job_data.get('version_control_experience', ['n/a']),
                "ci_cd_tools": job_data.get('ci_cd_tools', ['n/a']),
                "collaborative_tools": job_data.get('collaborative_tools', ['n/a']),
                "leadership_requirement": job_data.get('leadership_requirement', ['n/a']),
                "education_requirement": job_data.get('education_requirement', 'n/a'),
                "advanced_degree_preference": job_data.get('advanced_degree_preference', 'n/a'),
                "papers_publications_preferred": job_data.get('papers_publications_preferred', 'n/a'),
                "prior_startup_experience": job_data.get('prior_startup_experience', ['n/a']),
                "advancement_history_required": job_data.get('advancement_history_required', False),
                "independent_work_capacity": job_data.get('independent_work_capacity', 'n/a'),
                "skills_must_have": job_data.get('skills_must_have', ['n/a']),
                "skills_preferred": job_data.get('skills_preferred', ['n/a']),
                "product_details": job_data.get('product_details', 'n/a'),
                "product_development_stage": job_data.get('product_development_stage', ['n/a']),
                "technical_challenges": job_data.get('technical_challenges', ['n/a']),
                "key_responsibilities": job_data.get('key_responsibilities', ['n/a']),
                "scope_of_impact": job_data.get('scope_of_impact', ['n/a']),
                "expected_deliverables": job_data.get('expected_deliverables', ['n/a']),
                "product_development_methodology": job_data.get('product_development_methodology', ['n/a']),
                "stage_of_codebase": job_data.get('stage_of_codebase', ['n/a']),
                "growth_trajectory": job_data.get('growth_trajectory', 'n/a'),
                "founder_background": job_data.get('founder_background', 'n/a'),
                "funding_stability": job_data.get('funding_stability', 'n/a'),
                "expected_hours": job_data.get('expected_hours', 'n/a'),
                "ideal_companies": job_data.get('ideal_companies', ['n/a']),
                "deal_breakers": job_data.get('deal_breakers', ['n/a']),
                "culture_fit_indicators": job_data.get('culture_fit_indicators', ['n/a']),
                "startup_mindset_requirements": job_data.get('startup_mindset_requirements', ['n/a']),
                "autonomy_level_required": job_data.get('autonomy_level_required', 'n/a'),
                "growth_mindset_indicators": job_data.get('growth_mindset_indicators', ['n/a']),
                "ideal_candidate_profile": job_data.get('ideal_candidate_profile', 'n/a'),
                "interview_process_tags": job_data.get('interview_process_tags', ['n/a']),
                "technical_assessment_type": job_data.get('technical_assessment_type', ['n/a']),
                "interview_focus_areas": job_data.get('interview_focus_areas', ['n/a']),
                "time_to_hire": job_data.get('time_to_hire', 'n/a'),
                "decision_makers": job_data.get('decision_makers', ['n/a']),
                "recruiter_pitch_points": job_data.get('recruiter_pitch_points', ['n/a'])
            }
            
            # Store the job in Pinecone
            self.jobs_index.upsert(
                vectors=[{
                    "id": job_id,
                    "values": embedding,
                    "metadata": metadata
                }]
            )
            
            # Add a small delay to allow for indexing
            time.sleep(1)
            
            # Verify the job was stored by querying it
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    # Query the job by ID
                    query_response = self.jobs_index.query(
                        id=job_id,
                        top_k=1,
                        include_metadata=True
                    )
                    
                    if query_response.matches and query_response.matches[0].id == job_id:
                        logger.info(f"Successfully verified job storage for ID: {job_id}")
                        return {"status": "success", "job_id": job_id}
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Verification attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
                except Exception as e:
                    logger.error(f"Error verifying job storage (attempt {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
            
            # If we get here, verification failed but the job was likely stored
            logger.warning(f"Job {job_id} was likely stored but verification failed after {max_retries} attempts")
            return {"status": "success", "job_id": job_id, "verification": "incomplete"}
            
        except Exception as e:
            logger.error(f"Error storing job {job_id}: {str(e)}")
            raise 