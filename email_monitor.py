from fastapi import FastAPI, Request, HTTPException
from app.agents.interaction_agent import InteractionAgent
from typing import Dict, Any
import json
import re
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
interaction_agent = InteractionAgent()

def extract_email_thread_info(email_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant information from the email thread."""
    subject = email_data.get('subject', '')
    from_email = email_data.get('from', '')
    to_email = email_data.get('to', '')
    
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
    
    # Extract email content
    text_content = email_data.get('text', '')
    html_content = email_data.get('html', '')
    
    # Use HTML content if available, otherwise use text content
    email_content = html_content if html_content else text_content
    
    return {
        'subject': subject,
        'from_email': from_email,
        'to_email': to_email,
        'job_id': job_id,
        'candidate_id': candidate_id,
        'email_content': email_content,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.post("/email/webhook")
async def receive_email(request: Request):
    """Handle incoming emails from an email webhook (e.g., Gmail push notifications or other provider)."""
    try:
        # Parse the incoming email data
        form_data = await request.form()
        
        # Log the raw form data for debugging
        print("Received webhook data:", form_data)
        
        # Extract email data from form
        email_data = json.loads(form_data.get('email', '{}'))
        
        # Extract thread information
        thread_info = extract_email_thread_info(email_data)
        
        # Log the extracted thread info
        print("Extracted thread info:", thread_info)
        
        # If we can't identify the thread, try to process as a new conversation
        if not thread_info['job_id'] or not thread_info['candidate_id']:
            print("No thread IDs found, processing as new conversation")
            # Here you could implement logic to handle new conversations
            return {
                'status': 'success',
                'message': 'Email received but no thread IDs found',
                'thread_info': thread_info
            }
        
        # Process the reply using the interaction agent
        result = interaction_agent.handle_candidate_reply(
            thread_info['candidate_id'],
            thread_info['email_content'],
            thread_info['job_id']
        )
        
        return {
            'status': 'success',
            'thread_info': thread_info,
            'processing_result': result
        }
        
    except Exception as e:
        print(f"Error processing incoming email: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 