from typing import Dict, Any, List
import logging
from datetime import datetime
from app.config.supabase import get_supabase_client
from app.services.pinecone_service import PineconeService
from app.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

class MatchingService:
    def __init__(self):
        self.supabase = get_supabase_client()
        self.pinecone_service = PineconeService()
        self.openai_service = OpenAIService()

    async def match_candidate(self, candidate_id: str) -> List[Dict[str, Any]]:
        """
        Match a candidate with potential jobs using the intake matching agent.
        """
        try:
            logger.info(f"Starting matching process for candidate {candidate_id}")
            
            # 1. Get candidate profile from Supabase
            candidate = await self.supabase.table('candidates_dev').select('*').eq('id', candidate_id).single().execute()
            if not candidate.data:
                raise Exception(f"Candidate {candidate_id} not found")

            # 2. Generate embedding for candidate profile
            profile_text = self.openai_service._prepare_text_for_embedding(candidate.data.get('profile_json', {}))
            query_vector = await self.openai_service.generate_embedding(profile_text)

            # 3. Query Pinecone for matching jobs
            matches = await self.pinecone_service.query_matches(
                query_vector=query_vector,
                top_k=10,
                namespace='jobs'
            )

            # 4. Process matches and create match records
            match_records = []
            for match in matches:
                job_id = match.metadata.get('job_id')
                if not job_id:
                    continue

                # Calculate match score and reason
                match_score = match.score
                match_reason = self._generate_match_reason(candidate.data, match.metadata)

                # Create match record
                match_record = {
                    'id': str(uuid.uuid4()),
                    'candidate_id': candidate_id,
                    'job_id': job_id,
                    'match_score': match_score,
                    'match_reason': match_reason,
                    'match_tags': self._extract_match_tags(candidate.data, match.metadata),
                    'status': 'pending',
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }

                # Insert match record
                await self.supabase.table('candidate_job_matches_dev').insert(match_record).execute()
                match_records.append(match_record)

            logger.info(f"Created {len(match_records)} matches for candidate {candidate_id}")
            return match_records

        except Exception as e:
            logger.error(f"Error matching candidate {candidate_id}: {str(e)}")
            raise Exception(f"Error matching candidate: {str(e)}")

    def _generate_match_reason(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> str:
        """
        Generate a human-readable reason for the match.
        """
        reasons = []

        # Check tech stack match
        candidate_tech = set(candidate.get('profile_json', {}).get('tech_stack', []))
        job_tech = set(job.get('tech_stack', '').split(','))
        tech_overlap = candidate_tech.intersection(job_tech)
        if tech_overlap:
            reasons.append(f"Strong technical alignment with {len(tech_overlap)} matching technologies")

        # Check experience match
        candidate_exp = candidate.get('profile_json', {}).get('years_of_experience', 0)
        job_min_exp = job.get('minimum_years_of_experience', 0)
        if candidate_exp >= job_min_exp:
            reasons.append(f"Meets experience requirements ({candidate_exp} years)")

        # Check location match
        candidate_locations = set(candidate.get('profile_json', {}).get('work_preferences', {}).get('location', []))
        job_locations = set(job.get('location', {}).get('city', []))
        if candidate_locations.intersection(job_locations):
            reasons.append("Location preferences align")

        # Check work arrangement match
        candidate_arrangement = set(candidate.get('profile_json', {}).get('work_preferences', {}).get('arrangement', []))
        job_arrangement = set(job.get('work_arrangement', '').split(','))
        if candidate_arrangement.intersection(job_arrangement):
            reasons.append("Work arrangement preferences match")

        return " | ".join(reasons) if reasons else "General profile match"

    def _extract_match_tags(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> List[str]:
        """
        Extract relevant tags for the match.
        """
        tags = []

        # Tech stack tags
        candidate_tech = set(candidate.get('profile_json', {}).get('tech_stack', []))
        job_tech = set(job.get('tech_stack', '').split(','))
        tech_overlap = candidate_tech.intersection(job_tech)
        if tech_overlap:
            tags.extend(list(tech_overlap))

        # Experience level tag
        candidate_exp = candidate.get('profile_json', {}).get('years_of_experience', 0)
        job_min_exp = job.get('minimum_years_of_experience', 0)
        if candidate_exp >= job_min_exp:
            tags.append(f"{candidate_exp}+ years experience")

        # Location tags
        candidate_locations = set(candidate.get('profile_json', {}).get('work_preferences', {}).get('location', []))
        job_locations = set(job.get('location', {}).get('city', []))
        location_overlap = candidate_locations.intersection(job_locations)
        if location_overlap:
            tags.extend([f"Location: {loc}" for loc in location_overlap])

        # Work arrangement tags
        candidate_arrangement = set(candidate.get('profile_json', {}).get('work_preferences', {}).get('arrangement', []))
        job_arrangement = set(job.get('work_arrangement', '').split(','))
        arrangement_overlap = candidate_arrangement.intersection(job_arrangement)
        if arrangement_overlap:
            tags.extend([f"Arrangement: {arr}" for arr in arrangement_overlap])

        return tags 