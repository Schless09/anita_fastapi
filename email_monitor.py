from fastapi import FastAPI, Request
from agents.interaction_agent import InteractionAgent
from typing import Dict, Any
import json
import re
from datetime import datetime

app = FastAPI()
interaction_agent = InteractionAgent()

def extract_email_thread_info(email_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant information from the email thread."""
    subject = email_data.get('subject', '')
    from_email = email_data.get('from', '')
    
    # Extract candidate ID and job ID from subject or thread
    job_id_match = re.search(r'job_id[=:](\w+)', subject)
    candidate_id_match = re.search(r'candidate_id[=:](\w+)', subject)
    
    job_id = job_id_match.group(1) if job_id_match else None
    candidate_id = candidate_id_match.group(1) if candidate_id_match else None
    
    # If IDs not found in subject, try to extract from email headers or thread
    if not job_id or not candidate_id:
        headers = email_data.get('headers', {})
        references = headers.get('References', '')
        
        # Try to extract from references
        job_id = job_id or re.search(r'job_id[=:](\w+)', references)
        candidate_id = candidate_id or re.search(r'candidate_id[=:](\w+)', references)
    
    return {
        'subject': subject,
        'from_email': from_email,
        'job_id': job_id,
        'candidate_id': candidate_id,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.post("/email/webhook")
async def handle_incoming_email(request: Request):
    """Handle incoming emails from SendGrid webhook."""
    try:
        # Parse the incoming email data
        form_data = await request.form()
        email_data = json.loads(form_data.get('email'))
        
        # Extract thread information
        thread_info = extract_email_thread_info(email_data)
        
        # If we can't identify the thread, log and skip
        if not thread_info['job_id'] or not thread_info['candidate_id']:
            print(f"Could not identify thread for email: {thread_info}")
            return {'status': 'error', 'message': 'Could not identify email thread'}
        
        # Get the email content
        email_content = email_data.get('text', '')
        
        # Process the reply using the interaction agent
        result = interaction_agent.handle_candidate_reply(
            thread_info['candidate_id'],
            email_content,
            thread_info['job_id']
        )
        
        return {
            'status': 'success',
            'thread_info': thread_info,
            'processing_result': result
        }
        
    except Exception as e:
        print(f"Error processing incoming email: {str(e)}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 