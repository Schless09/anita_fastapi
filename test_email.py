import os
import ssl
import logging
from agents.interaction_agent import InteractionAgent
from config.sendgrid import SENDGRID_CONFIG
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_job_match_email():
    # Force reload environment variables
    load_dotenv(override=True)
    
    # Debug environment variables
    logger.info("=== Environment Variables ===")
    logger.info(f"SENDER_EMAIL from env: {os.getenv('SENDER_EMAIL')}")
    logger.info(f"SENDER_EMAIL from config: {SENDGRID_CONFIG['sender_email']}")
    logger.info("=== End Environment Variables ===")
    
    # Log environment variables (masking sensitive data)
    logger.info("Checking environment variables...")
    sendgrid_key = SENDGRID_CONFIG['api_key']
    sender_email = SENDGRID_CONFIG['sender_email']
    logger.info(f"SendGrid API Key present: {'Yes' if sendgrid_key else 'No'}")
    logger.info(f"Sender Email: {sender_email}")
    
    # Disable SSL verification for testing
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

    # Create a test job match with realistic data
    job_match = {
        'email': 'harrison.franke@gmail.com',
        'match_score': 95.5,
        'match_reason': 'Your experience with Python, FastAPI, and building scalable backend systems aligns perfectly with this role. Your work on distributed systems and API development matches the technical requirements. Your experience with modern cloud technologies and containerization is exactly what we\'re looking for.',
        'job_details': {
            'job_title': 'Senior Backend Engineer',
            'company_name': 'Hedra',
            'location_city': 'San Francisco',
            'location_state': 'CA',
            'description': 'We are looking for a Senior Backend Engineer to join our team and help build scalable, high-performance systems. The ideal candidate will have strong experience with Python, FastAPI, and distributed systems. You\'ll be working on our core platform, building APIs that serve millions of requests daily.',
            'requirements': '- 5+ years of experience with Python and modern backend frameworks\n- Strong experience with FastAPI and building RESTful APIs\n- Experience with distributed systems and microservices\n- Knowledge of cloud platforms (AWS, GCP, or Azure)\n- Experience with containerization and Kubernetes\n- Strong understanding of system design and scalability',
            'benefits': '- Competitive salary and equity\n- Comprehensive health insurance\n- Remote-first culture\n- Flexible work hours\n- Professional development budget\n- Modern tech stack and tools\n- Collaborative team environment',
            'paraform_link': 'https://www.paraform.com/share/hedra/cm2pcqjin002il90czttxt1sy'
        }
    }

    logger.info("Initializing InteractionAgent...")
    agent = InteractionAgent()
    
    logger.info("Attempting to send email...")
    result = agent.contact_candidate(job_match)
    
    logger.info(f"Email sending result: {result}")
    return result

if __name__ == "__main__":
    test_job_match_email() 