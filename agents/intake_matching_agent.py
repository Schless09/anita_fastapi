# agents/intake_matching_agent.py
from .vector_store import VectorStore
from typing import Dict, Any, Optional, List
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IntakeMatchingAgent:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize IntakeMatchingAgent with an optional vector_store."""
        self.vector_store = vector_store or VectorStore()

    def match_candidate(self, candidate_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Match a candidate with job opportunities using vector similarity and dealbreakers.
        Returns the best match that passes all dealbreaker criteria.
        """
        # Store/update candidate in vector database
        candidate_id = candidate_data.get('id', 'default_id')
        logger.info(f"\n🔍 Starting job matching process for candidate {candidate_id}")
        
        # Store candidate in vector database
        logger.info(f"💾 Storing candidate data in vector database...")
        self.vector_store.store_candidate(candidate_id, candidate_data)
        
        # Find matching jobs with enhanced matching
        logger.info(f"🔎 Searching for matching jobs...")
        matches = self.vector_store.find_similar_jobs(candidate_id)
        
        if matches['status'] == 'success' and matches['matches']:
            best_match = matches['matches'][0]  # First match is highest scoring
            match_score = best_match['score']
            job_title = best_match['metadata'].get('job_title', 'N/A')
            company = best_match['metadata'].get('company_name', 'N/A')
            
            logger.info(f"🎯 Found {len(matches['matches'])} potential matches")
            logger.info(f"⭐ Best match: {job_title} at {company}")
            logger.info(f"📊 Match score: {match_score:.2f}")
            
            # Log dealbreakers
            dealbreakers = best_match.get('dealbreakers', {})
            if dealbreakers:
                logger.info("✅ Dealbreakers check:")
                for key, value in dealbreakers.items():
                    status = "✅" if value else "❌"
                    logger.info(f"  {status} {key.replace('_', ' ').title()}")
            
            return {
                'candidate_id': candidate_id,
                'job_id': best_match['job_id'],
                'match_score': best_match['score'],
                'job_details': best_match['metadata'],
                'dealbreakers': best_match.get('dealbreakers'),
                'match_reason': best_match.get('match_reason'),
                'phone_number': candidate_data.get('phone_number'),
                'email': candidate_data.get('email')
            }
        else:
            logger.warning(f"⚠️ No matches found for candidate {candidate_id}")
            return None

    def fetch_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open positions from the jobs index"""
        try:
            # Query all vectors in the jobs index
            # Note: In a production environment, you might want to paginate this
            response = self.vector_store.jobs_index.query(
                vector=[0] * 1536,  # Dummy vector for OpenAI ada-002 dimension
                top_k=100,  # Adjust based on your needs
                include_metadata=True
            )
            
            return [
                {
                    'job_id': match['id'],
                    'metadata': match['metadata']
                }
                for match in (response['matches'] or [])
                if match['metadata']
            ]
        except Exception as e:
            print(f"Error fetching open positions: {str(e)}")
            return []

    def find_candidates_for_job(self, job_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find candidates that match a specific job posting using enhanced matching"""
        return self.vector_store.find_similar_candidates(job_id, top_k)