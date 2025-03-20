import os
from pinecone import Pinecone
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
import time
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

class VectorStore:
    def __init__(self, init_openai: bool = False):
        """Initialize Pinecone with environment variables."""
        print("Initializing Pinecone...")
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        print("Pinecone initialized successfully")
        print("Attempting to connect to indexes: anita-candidates, job-details")
        
        # Get list of existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        print("Existing indexes:", existing_indexes)
        
        # Connect to indexes
        self.candidates_index = self.pc.Index("anita-candidates")
        self.jobs_index = self.pc.Index("job-details")
        
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
                "linkedin": candidate_data.get("linkedin"),
                "candidate_id": candidate_id,  # Add this for easier querying
                "timestamp": datetime.utcnow().isoformat()
            }
            
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

    def find_similar_jobs(self, candidate_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find jobs similar to a candidate's profile."""
        try:
            print(f"\n=== Finding jobs for candidate {candidate_id} ===")
            # First try to find the candidate using the candidate_id in metadata
            print("Searching by candidate_id in metadata...")
            candidate_chunks = self.candidates_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata query
                filter={"candidate_id": candidate_id},  # Search by candidate_id in metadata
                include_metadata=True,
                top_k=10  # Get all chunks
            )
            
            if not candidate_chunks.matches:
                print("No matches found by candidate_id in metadata, trying exact ID match...")
                # Try exact ID match as fallback
                candidate_chunks = self.candidates_index.query(
                    vector=[0] * 1536,
                    filter={"id": candidate_id},  # Try exact match
                    include_metadata=True,
                    top_k=1
                )
                
                if not candidate_chunks.matches:
                    print("No exact match found, trying with chunk suffix...")
                    # Try with chunk suffix as last resort
                    candidate_chunks = self.candidates_index.query(
                        vector=[0] * 1536,
                        filter={"id": f"{candidate_id}_chunk_0"},  # Try first chunk
                        include_metadata=True,
                        top_k=1
                    )
            
            if not candidate_chunks.matches:
                print(f"No matches found for candidate {candidate_id}")
                return {
                    "status": "error",
                    "message": f"Candidate {candidate_id} not found"
                }
            
            print(f"Found {len(candidate_chunks.matches)} matches")
            # Use the first chunk's vector for job matching (usually contains the most relevant info)
            first_match = candidate_chunks.matches[0]
            print(f"Using vector from: {first_match.id}")
            
            # Query jobs index with the candidate's vector
            print("\nQuerying jobs index...")
            results = self.jobs_index.query(
                vector=first_match.values,
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
            
            print(f"Found {len(matches)} job matches")
            return {
                "status": "success",
                "matches": matches,
                "candidate_id": first_match.metadata.get("candidate_id", first_match.id),
                "candidate_metadata": first_match.metadata
            }
            
        except Exception as e:
            print(f"Error finding similar jobs: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            print("=== Job search complete ===\n")

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