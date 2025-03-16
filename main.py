from fastapi import FastAPI
from agents.brain_agent import BrainAgent
from agents.interaction_agent import InteractionAgent

app = FastAPI()
interaction_agent = InteractionAgent()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/test-sms")
async def test_sms():
    # Test data for Andrew
    job_match = {
        'phone_number': '+18476094515',
        'title': 'Senior Backend Engineer',
        'company': 'Hedra'
    }
    
    result = interaction_agent.contact_candidate(job_match)
    return result

