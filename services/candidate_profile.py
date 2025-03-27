from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from supabase import Client
from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CandidateProfile(BaseModel):
    supabase_id: str
    pinecone_id: str
    email: str
    name: Optional[str]
    phone: Optional[str]
    vector_embeddings: Optional[Dict[str, List[float]]]
    structured_data: Dict[str, Any]
    last_updated: datetime
    last_sync: datetime

class CandidateProfileService:
    def __init__(self, supabase: Client, pinecone_client: Pinecone, pinecone_index: str, table_prefix: str = 'dev'):
        self.supabase = supabase
        self.pc = pinecone_client
        self.pinecone_index = pinecone_index
        self.table_prefix = table_prefix
        self.openai = OpenAI()
        self.index = self.pc.Index(self.pinecone_index)
    
    def _table(self, name: str) -> str:
        """Get the table name with the appropriate environment suffix"""
        return f"{name}-{self.table_prefix}"
    
    async def get_or_create_candidate(self, email: str, phone: Optional[str] = None, name: Optional[str] = None) -> CandidateProfile:
        """Get or create a candidate profile with data consistency across Supabase and Pinecone."""
        try:
            # Check Supabase first
            result = self.supabase.table(self._table('candidates')).select('*').eq('email', email).execute()
            
            if result.data:
                candidate_data = result.data[0]
                # Update if new info provided
                if phone or name:
                    updates = {}
                    if phone:
                        updates['phone'] = phone
                    if name:
                        updates['name'] = name
                    if updates:
                        self.supabase.table(self._table('candidates')).update(updates).eq('id', candidate_data['id']).execute()
                        candidate_data.update(updates)
            else:
                # Create new candidate in Supabase
                candidate_data = self.supabase.table(self._table('candidates')).insert({
                    'email': email,
                    'phone': phone,
                    'name': name,
                    'created_at': datetime.utcnow().isoformat(),
                    'last_sync': datetime.utcnow().isoformat()
                }).execute().data[0]
            
            # Ensure Pinecone consistency
            await self.sync_with_pinecone(candidate_data['id'])
            
            return await self.get_unified_profile(candidate_data['id'])
            
        except Exception as e:
            logger.error(f"Error in get_or_create_candidate: {str(e)}")
            raise
    
    async def sync_with_pinecone(self, candidate_id: str):
        """Ensure Pinecone has the latest candidate data."""
        try:
            # Get all candidate data from Supabase
            candidate = self.supabase.table(self._table('candidates')).select('*').eq('id', candidate_id).execute().data[0]
            
            # Get associated data
            resume = self.supabase.table(self._table('resumes')).select('*').eq('candidate_id', candidate_id).execute().data
            calls = self.supabase.table(self._table('calls')).select('*').eq('candidate_id', candidate_id).execute().data
            skills = self.supabase.table(self._table('candidate_skills')).select('*').eq('candidate_id', candidate_id).execute().data
            
            # Create comprehensive text for embedding
            profile_text = f"""
            Candidate Name: {candidate.get('name', '')}
            Email: {candidate.get('email', '')}
            Phone: {candidate.get('phone', '')}
            
            Resume: {resume[0].get('raw_text', '') if resume else ''}
            
            Skills: {', '.join(skill['name'] for skill in skills)}
            
            Interview Transcripts:
            {' '.join(call.get('transcript', '') for call in calls if call.get('transcript'))}
            """
            
            # Generate embedding
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=profile_text
            )
            vector = response.data[0].embedding
            
            # Update Pinecone
            metadata = {
                'candidate_id': candidate_id,
                'email': candidate['email'],
                'last_updated': datetime.utcnow().isoformat(),
                'has_resume': bool(resume),
                'call_count': len(calls),
                'skill_count': len(skills)
            }
            
            self.index.upsert(
                vectors=[(candidate_id, vector, metadata)]
            )
            
            # Update last sync timestamp in Supabase
            self.supabase.table(self._table('candidates')).update({
                'last_sync': datetime.utcnow().isoformat()
            }).eq('id', candidate_id).execute()
            
            logger.info(f"Successfully synced candidate {candidate_id} with Pinecone")
            
        except Exception as e:
            logger.error(f"Error in sync_with_pinecone: {str(e)}")
            raise
    
    async def get_unified_profile(self, candidate_id: str) -> CandidateProfile:
        """Get a unified view of the candidate's profile from both Supabase and Pinecone."""
        try:
            # Get Supabase data
            candidate = self.supabase.table(self._table('candidates')).select('*').eq('id', candidate_id).execute().data[0]
            
            # Get Pinecone data
            vector_data = self.index.fetch(ids=[candidate_id])
            
            vector_info = vector_data.vectors.get(candidate_id, {})
            
            return CandidateProfile(
                supabase_id=candidate_id,
                pinecone_id=candidate_id,
                email=candidate['email'],
                name=candidate.get('name'),
                phone=candidate.get('phone'),
                vector_embeddings={
                    'profile': vector_info.get('values', [])
                },
                structured_data={
                    'supabase': candidate,
                    'pinecone_metadata': vector_info.get('metadata', {})
                },
                last_updated=datetime.fromisoformat(candidate['updated_at']),
                last_sync=datetime.fromisoformat(candidate['last_sync'])
            )
            
        except Exception as e:
            logger.error(f"Error in get_unified_profile: {str(e)}")
            raise
    
    async def find_matching_jobs(self, candidate_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find jobs that match the candidate's profile."""
        try:
            # Get candidate vector
            vector_data = self.index.fetch(ids=[candidate_id])
            
            if not vector_data.vectors:
                await self.sync_with_pinecone(candidate_id)
                vector_data = self.index.fetch(ids=[candidate_id])
            
            if not vector_data.vectors:
                raise Exception(f"Could not find vector for candidate {candidate_id}")
            
            vector = vector_data.vectors[candidate_id].values
            
            # Query job index
            try:
                job_index = self.pc.Index('job-details')
                matches = job_index.query(
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True
                )
                
                return [
                    {
                        'job_id': match.id,
                        'score': match.score,
                        'metadata': match.metadata
                    }
                    for match in matches.matches
                ]
            except Exception as e:
                logger.warning(f"Failed to query job index: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"Error in find_matching_jobs: {str(e)}")
            raise 