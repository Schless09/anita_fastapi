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
from .matching_utils import (
    check_location_match,
    check_work_environment_match,
    check_compensation_match,
    check_work_authorization_match,
    generate_match_reason
)

class VectorStore:
    def __init__(self, init_openai: bool = False):
        """Initialize Pinecone with environment variables."""
        print("Initializing Pinecone...")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get index names from environment variables with defaults
        candidates_index_name = os.getenv('PINECONE_INDEX_NAME', 'anita-candidates-dev')
        jobs_index_name = os.getenv('PINECONE_INDEX_NAME_JOBS', 'job-details')
        
        print(f"Pinecone initialized successfully")
        print(f"Attempting to connect to indexes: {candidates_index_name}, {jobs_index_name}")
        
        # Get list of existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        print("Existing indexes:", existing_indexes)
        
        # Connect to indexes
        self.candidates_index = self.pc.Index(candidates_index_name)
        self.jobs_index = self.pc.Index(jobs_index_name)
        
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

    def store_job(self, job_id: str, job_data: Dict[str, Any]) -> Dict[str, str]:
        """Store job posting in vector database."""
        try:
            print(f"\n=== Storing job {job_id} ===")
            
            # Validate required fields
            required_fields = ['job_title', 'company_name', 'paraform_link']
            missing_fields = [field for field in required_fields if not job_data.get(field)]
            if missing_fields:
                print(f"Missing required fields: {', '.join(missing_fields)}")
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Create text representation for embedding
            job_text = f"""
            Title: {job_data.get('job_title', '')}
            Company: {job_data.get('company_name', '')}
            Location: {job_data.get('location_city', '')}, {job_data.get('location_state', '')}
            Description: {job_data.get('description', '')}
            Requirements: {job_data.get('requirements', '')}
            Benefits: {job_data.get('benefits', '')}
            """
            
            # Create embedding for the job
            vector = self.get_embedding(job_text)
            
            # Prepare metadata
            metadata = {
                "job_id": job_id,
                "job_title": job_data.get("job_title"),
                "company_name": job_data.get("company_name"),
                "location_city": job_data.get("location_city"),
                "location_state": job_data.get("location_state"),
                "description": job_data.get("description"),
                "requirements": job_data.get("requirements"),
                "benefits": job_data.get("benefits"),
                "paraform_link": job_data.get("paraform_link"),
                "role_details": job_data.get("role_details", {}),
                "company_information": job_data.get("company_information", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Pinecone
            print("Storing job in Pinecone...")
            self.jobs_index.upsert(vectors=[(
                job_id,
                vector,
                metadata
            )])
            
            # Verify storage
            print("\nVerifying storage...")
            verification = self.jobs_index.query(
                vector=[0] * 1536,
                filter={"job_id": job_id},
                top_k=1,
                include_metadata=True
            )
            
            if verification.matches:
                print(f"Successfully verified storage. Found job with ID: {verification.matches[0].id}")
            else:
                print("Warning: Could not verify storage immediately. This might be due to indexing delay.")
            
            return {
                "status": "success",
                "message": f"Job {job_id} stored successfully"
            }
            
        except Exception as e:
            print(f"Error storing job: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            print("=== Store job operation complete ===\n") 