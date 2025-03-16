# agents/brain_agent.py
from .intake_agent import IntakeAgent
from .intake_matching_agent import IntakeMatchingAgent
from .interaction_agent import InteractionAgent

class BrainAgent:
    def __init__(self):
        self.intake_agent = IntakeAgent()
        self.matching_agent = IntakeMatchingAgent()
        self.interaction_agent = InteractionAgent()
        self.state = {}  # Initialize state dictionary

    def handle_candidate_submission(self, candidate_data):
        """Handles new candidate submissions."""
        candidate_id = candidate_data.get('id', 'default_id')  # Provide default ID if none exists
        self.state[candidate_id] = 'screening'
        
        screening_result = self.intake_agent.screen_candidate(candidate_data)
        if screening_result:
            self.state[candidate_id] = 'matching'
            match_result = self.matching_agent.match_candidate(screening_result)
            if match_result:
                self.state[candidate_id] = 'interaction'
                self.interaction_agent.contact_candidate(match_result)
                self.state[candidate_id] = 'completed'
            else:
                self.state[candidate_id] = 'store_for_future'
                self.store_for_future_opportunities(candidate_data)
        else:
            self.state[candidate_id] = 'screening_failed'

    def store_for_future_opportunities(self, candidate_data):
        # Logic to store candidate for future opportunities
        pass