# agents/brain_agent.py
from .intake_agent import IntakeAgent
from .intake_matching_agent import IntakeMatchingAgent
from .interaction_agent import InteractionAgent
from .vector_store import VectorStore
from typing import Dict, Any, Optional

class BrainAgent:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.intake_agent = IntakeAgent()
        self.matching_agent = IntakeMatchingAgent(self.vector_store)
        self.interaction_agent = InteractionAgent()
        self.state = {}  # Initialize state dictionary
        self.candidate_profiles = {}  # Store candidate profiles with additional data

    def handle_candidate_submission(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles new candidate submissions with enhanced matching."""
        candidate_id = candidate_data.get('id', 'default_id')
        self.state[candidate_id] = 'screening'
        
        # Initialize or update candidate profile
        if candidate_id not in self.candidate_profiles:
            self.candidate_profiles[candidate_id] = {
                'basic_info': candidate_data,
                'transcript': None,
                'screening_result': None,
                'match_result': None,
                'vector_id': None,
                'dealbreakers': None,
                'match_reason': None
            }
        
        # Store candidate in vector database with enhanced metadata
        vector_result = self.vector_store.store_candidate(candidate_id, candidate_data)
        self.candidate_profiles[candidate_id]['vector_id'] = candidate_id
        
        screening_result = self.intake_agent.screen_candidate(candidate_data)
        if screening_result:
            self.state[candidate_id] = 'matching'
            self.candidate_profiles[candidate_id]['screening_result'] = screening_result
            
            match_result = self.matching_agent.match_candidate(screening_result)
            if match_result:
                self.state[candidate_id] = 'interaction'
                self.candidate_profiles[candidate_id]['match_result'] = match_result
                self.candidate_profiles[candidate_id]['dealbreakers'] = match_result.get('dealbreakers')
                self.candidate_profiles[candidate_id]['match_reason'] = match_result.get('match_reason')
                
                # Include enhanced matching information in the interaction
                self.interaction_agent.contact_candidate(match_result)
                self.state[candidate_id] = 'completed'
                
                return {
                    'status': 'success',
                    'message': 'Candidate processed successfully',
                    'state': 'completed',
                    'match_details': {
                        'job_id': match_result['job_id'],
                        'match_score': match_result['match_score'],
                        'match_reason': match_result['match_reason'],
                        'dealbreakers': match_result['dealbreakers']
                    }
                }
            else:
                self.state[candidate_id] = 'store_for_future'
                store_result = self.store_for_future_opportunities(candidate_data)
                return {
                    'status': 'success',
                    'message': 'No immediate matches found',
                    'state': 'store_for_future',
                    'store_result': store_result
                }
        else:
            self.state[candidate_id] = 'screening_failed'
            return {
                'status': 'error',
                'message': 'Candidate did not pass screening',
                'state': 'screening_failed'
            }

    def add_transcript_to_profile(self, candidate_id: str, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add processed transcript data to candidate profile and update vector store."""
        if candidate_id not in self.candidate_profiles:
            self.candidate_profiles[candidate_id] = {
                'basic_info': {},
                'transcript': None,
                'processed_transcript': None,
                'screening_result': None,
                'match_result': None,
                'vector_id': None,
                'dealbreakers': None,
                'match_reason': None
            }
        
        # Store both raw and processed transcript
        self.candidate_profiles[candidate_id]['transcript'] = transcript_data['raw_transcript']
        self.candidate_profiles[candidate_id]['processed_transcript'] = transcript_data['processed_data']
        
        # Update basic info with processed data
        processed_data = transcript_data['processed_data']
        basic_info = self.candidate_profiles[candidate_id]['basic_info']
        
        # Update candidate information with processed data
        if not basic_info.get('name') and processed_data.get('candidate_name'):
            basic_info['name'] = processed_data['candidate_name']
        
        if not basic_info.get('linkedin') and processed_data.get('linkedin_url'):
            basic_info['linkedin'] = processed_data['linkedin_url']
        
        if processed_data.get('contact_information'):
            if not basic_info.get('email') and processed_data['contact_information'].get('email'):
                basic_info['email'] = processed_data['contact_information']['email']
            if not basic_info.get('phone_number') and processed_data['contact_information'].get('phone'):
                basic_info['phone_number'] = processed_data['contact_information']['phone']
        
        if not basic_info.get('experience') and processed_data.get('years_of_experience'):
            basic_info['experience'] = str(processed_data['years_of_experience']) + " years"
        
        if processed_data.get('skills'):
            if not basic_info.get('skills'):
                basic_info['skills'] = processed_data['skills']
            else:
                # Merge skills lists while removing duplicates
                basic_info['skills'] = list(set(basic_info['skills'] + processed_data['skills']))
        
        if not basic_info.get('preferred_work_environment') and processed_data.get('preferred_work_environment'):
            basic_info['preferred_work_environment'] = processed_data['preferred_work_environment']
        
        if processed_data.get('preferred_locations'):
            if not basic_info.get('preferred_locations'):
                basic_info['preferred_locations'] = processed_data['preferred_locations']
        
        if not basic_info.get('minimum_salary') and processed_data.get('minimum_salary'):
            basic_info['minimum_salary'] = processed_data['minimum_salary']
        
        if not basic_info.get('work_authorization') and processed_data.get('work_authorization'):
            basic_info['work_authorization'] = processed_data['work_authorization']
        
        # Update candidate data in vector store
        candidate_data = basic_info.copy()
        candidate_data['transcript'] = transcript_data['raw_transcript']
        self.vector_store.store_candidate(candidate_id, candidate_data)
        
        # Re-evaluate candidate with new information if needed
        if self.state.get(candidate_id) in ['screening_failed', 'store_for_future']:
            return self.handle_candidate_submission(candidate_data)
        
        return {
            'status': 'success',
            'message': f'Transcript and processed data added to profile for candidate {candidate_id}',
            'current_state': self.state.get(candidate_id, 'unknown')
        }

    def store_for_future_opportunities(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store candidate data for future matching with enhanced metadata."""
        candidate_id = candidate_data.get('id', 'default_id')
        if candidate_id not in self.candidate_profiles:
            self.candidate_profiles[candidate_id] = {
                'basic_info': candidate_data,
                'transcript': None,
                'screening_result': None,
                'match_result': None,
                'vector_id': candidate_id,
                'dealbreakers': None,
                'match_reason': None
            }
            
        # Store candidate in vector database with enhanced metadata
        store_result = self.vector_store.store_candidate(candidate_id, candidate_data)
        
        return {
            'status': 'stored',
            'message': f'Candidate {candidate_id} stored for future opportunities',
            'store_result': store_result
        }

    def find_similar_candidates(self, job_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find candidates that match a specific job posting with enhanced matching."""
        return self.matching_agent.find_candidates_for_job(job_id, top_k)