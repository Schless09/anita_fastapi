from agents.brain_agent import BrainAgent
from agents.vector_store import VectorStore
from pprint import pprint

def test_matching():
    print("\n=== Testing Job-Candidate Matching ===")
    
    # Initialize agents
    brain = BrainAgent()
    vector_store = VectorStore()
    
    # Get open positions
    print('\n=== Available Jobs ===')
    positions = brain.matching_agent.fetch_open_positions()
    
    if not positions:
        print("No jobs found in the database!")
        return
        
    for pos in positions:
        job_metadata = pos.get('metadata', {})
        print(f"\nJob ID: {pos.get('job_id')}")
        print(f"Title: {job_metadata.get('job_title', 'N/A')}")
        print(f"Company: {job_metadata.get('company_name', 'N/A')}")
        print(f"Location: {job_metadata.get('location_city', 'N/A')}, {job_metadata.get('location_state', 'N/A')}")
        print(f"Seniority: {job_metadata.get('seniority_level', 'N/A')}")
        
        # Find matching candidates for this job
        print("\n--- Top Matching Candidates ---")
        matches = brain.find_similar_candidates(pos.get('job_id'), top_k=3)
        
        if matches.get('status') == 'success':
            candidates = matches.get('matches', [])
            if not candidates:
                print("No matching candidates found.")
                continue
                
            for match in candidates:
                candidate_metadata = match.get('metadata', {})
                print(f"\nCandidate ID: {match.get('candidate_id')}")
                print(f"Match Score: {match.get('score'):.2f}")
                print(f"Name: {candidate_metadata.get('name', 'N/A')}")
                print(f"Experience: {candidate_metadata.get('experience', 'N/A')}")
                print(f"Location: {candidate_metadata.get('location', 'N/A')}")
                
                # Show dealbreakers summary
                dealbreakers = match.get('dealbreakers', {})
                if dealbreakers:
                    print("\nDealbreakers Check:")
                    for check, result in dealbreakers.items():
                        print(f"- {check}: {'✓' if result else '✗'}")
        else:
            print(f"Error finding matches: {matches.get('message')}")
            
        print("\n" + "="*50)

if __name__ == "__main__":
    test_matching() 