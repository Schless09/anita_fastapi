import os
import logging
import ssl
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, From, Subject, HtmlContent, PlainTextContent
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from datetime import datetime
from config.sendgrid import SENDGRID_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

class InteractionAgent:
    def __init__(self):
        load_dotenv()
        self.sendgrid_api_key = SENDGRID_CONFIG['api_key']
        self.sender_email = SENDGRID_CONFIG['sender_email']
        
        # Handle SSL verification
        if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context
        
        # Log initialization details
        logger.info("Initializing InteractionAgent...")
        logger.info(f"SendGrid API Key present: {'Yes' if self.sendgrid_api_key else 'No'}")
        logger.info(f"Sender Email: {self.sender_email}")
        
        if not self.sendgrid_api_key:
            logger.error("SendGrid API key is missing!")
        if not self.sender_email:
            logger.error("Sender email is missing!")

        self.conversation_history: Dict[str, Dict[str, Any]] = {}

    def contact_candidate(self, job_match):
        """Sends an email to the candidate about the job opportunity. Be friendly and professional."""
        try:
            logger.info("Starting contact_candidate method...")
            recipient_email = job_match.get('email')
            job_details = job_match.get('job_details', {})
            job_title = job_details.get('job_title', 'Senior Backend Engineer')
            company = job_details.get('company_name', 'Hedra')
            match_score = job_match.get('match_score', 0)
            match_reason = job_match.get('match_reason', '')
            paraform_link = job_details.get('paraform_link', 'https://www.paraform.com/share/hedra/cm2pcqjin002il90czttxt1sy')

            logger.info(f"Preparing email for {recipient_email}")
            logger.info(f"Job details: {job_title} at {company}")

            # Create the email message
            message = Mail(
                from_email=self.sender_email,
                to_emails=recipient_email,
                subject=f"Job Opportunity: {job_title} at {company}",
                html_content=f"""
                <p>Hi there,</p>
                <p>I hope you're doing well! I came across an exciting opportunity that aligns with your profile and wanted to share it with you.</p>

                <p><strong>Role:</strong> {job_title}<br>
                <strong>Company:</strong> {company}<br>
                <strong>Match Score:</strong> {match_score:.2f}</p>

                <p><strong>Why this role matches your profile:</strong><br>
                {match_reason}</p>

                <p>You can find more details about the position and apply here:<br>
                <a href="{paraform_link}" target="_blank">View Job Details on Paraform</a></p>

                <p>Would you be interested in learning more? I'd be happy to hop on a quick call to discuss the role in detail and answer any questions you may have.</p>

                <p>If you'd like us to present your profile to the hiring manager, just reply with <strong>1-2 bullet points</strong> highlighting why you'd be a great fit, and I'll submit your application.</p>

                <p>Not interested? No worries! Let me know (along with <strong>1-2 reasons why</strong>)—this helps us refine our matching algorithm and ensure better opportunities for you in the future.</p>

                <p>Looking forward to your thoughts!</p>

                <p>Best,<br>
                <strong>Anita</strong><br>
                <em>Your AI Recruiter</em></p>
                """
            )

            logger.info("Creating SendGrid client...")
            sg = SendGridAPIClient(self.sendgrid_api_key)
            
            logger.info("Attempting to send email...")
            response = sg.send(message)
            
            logger.info(f"SendGrid response status code: {response.status_code}")
            logger.info(f"SendGrid response headers: {response.headers}")
            
            return {
                'status': 'success',
                'recipient': recipient_email,
                'subject': message.subject,
                'response_code': response.status_code
            }
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e),
                'recipient': recipient_email
            }

    def test_email(self):
        """Tests the email sending functionality."""
        recipient_email = "arschuessler90@gmail.com"

        message = Mail(
            from_email=self.sender_email,
            to_emails=recipient_email,
            subject="Test Email",
            html_content="<p>This is a test email from SendGrid.</p>"
        )

        try:
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            return {
                'status': 'success',
                'recipient': recipient_email,
                'subject': message.subject,
                'response_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'recipient': recipient_email
            }

    def handle_candidate_reply(self, candidate_id: str, email_content: str, job_id: str) -> Dict[str, Any]:
        """Process a candidate's reply to a job opportunity."""
        try:
            print(f"\n=== Processing reply from candidate {candidate_id} for job {job_id} ===")
            
            # Store the reply in conversation history
            if job_id not in self.conversation_history:
                self.conversation_history[job_id] = {
                    'candidate_id': candidate_id,
                    'replies': [],
                    'status': 'active'
                }
            
            # Add the new reply
            self.conversation_history[job_id]['replies'].append({
                'content': email_content,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Analyze the reply content
            is_interested = self._analyze_interest(email_content)
            
            if is_interested:
                # Send a follow-up email with next steps
                return self._send_follow_up(candidate_id, job_id, "interested")
            else:
                # Send a thank you email and ask for feedback
                return self._send_follow_up(candidate_id, job_id, "not_interested")
                
        except Exception as e:
            print(f"Error processing candidate reply: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _analyze_interest(self, email_content: str) -> bool:
        """Analyze the email content to determine if the candidate is interested."""
        # Convert to lowercase for case-insensitive matching
        content = email_content.lower()
        
        # Keywords indicating interest
        interest_keywords = [
            'interested', 'yes', 'sure', 'definitely', 'would love to',
            'sounds great', 'perfect match', 'good fit', 'apply',
            'submit', 'application', 'proceed', 'next steps'
        ]
        
        # Keywords indicating disinterest
        disinterest_keywords = [
            'not interested', 'no thanks', 'pass', 'decline',
            'not a good fit', 'not looking', 'not right now',
            'thanks but no', 'thank you but'
        ]
        
        # Count matches
        interest_count = sum(1 for keyword in interest_keywords if keyword in content)
        disinterest_count = sum(1 for keyword in disinterest_keywords if keyword in content)
        
        # If we have more interest keywords than disinterest keywords, consider them interested
        return interest_count > disinterest_count

    def _send_follow_up(self, candidate_id: str, job_id: str, interest_status: str) -> Dict[str, Any]:
        """Send a follow-up email based on the candidate's interest status."""
        try:
            # Get job details from conversation history
            job_details = self.conversation_history.get(job_id, {})
            
            if interest_status == "interested":
                subject = "Great to hear you're interested!"
                content = f"""
                <p>Thank you for your interest in the position! I'm excited to help you move forward.</p>
                
                <p>Next steps:</p>
                <ol>
                    <li>I'll review your response and prepare your application</li>
                    <li>I'll submit your profile to the hiring team</li>
                    <li>I'll keep you updated on the process</li>
                </ol>
                
                <p>Is there anything specific you'd like to know about the role or the company?</p>
                
                <p>Best regards,<br>
                <strong>Anita</strong><br>
                <em>Your AI Recruiter</em></p>
                """
            else:
                subject = "Thank you for your response"
                content = f"""
                <p>Thank you for taking the time to respond. I appreciate your feedback!</p>
                
                <p>To help me find better opportunities for you in the future, could you share:</p>
                <ul>
                    <li>What aspects of the role weren't a good fit?</li>
                    <li>What kind of opportunities are you looking for?</li>
                </ul>
                
                <p>Best regards,<br>
                <strong>Anita</strong><br>
                <em>Your AI Career Co-Pilot</em></p>
                """
            
            # Create and send the email
            message = Mail(
                from_email=self.sender_email,
                to_emails=candidate_id,  # This should be the candidate's email
                subject=subject,
                html_content=content
            )
            
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            
            return {
                'status': 'success',
                'message': f'Follow-up email sent for {interest_status} response',
                'response_code': response.status_code
            }
            
        except Exception as e:
            print(f"Error sending follow-up email: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def send_transcript_summary(self, email: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a summary email to a candidate with their transcript analysis."""
        try:
            logger.info(f"Preparing transcript summary email for {email}")
            
            # Create SendGrid client

            # Create the email message
            message = Mail(
                from_email=Email(self.sender_email),
                to_emails=To(email),
                subject="Thanks for speaking with me!",
                html_content=f"""
                <p>Hi {processed_data.get('first_name', '')},</p>
                
                <p>Thank you for taking the time to speak with me today! I wanted to summarize the key points from our conversation:</p>

                <h3>Key Points from Our Conversation:</h3>
                <ul>
                    {"".join(f"<li>{point}</li>" for point in processed_data.get('key_points', []))}
                </ul>

                <h3>Your Experience Highlights:</h3>
                <ul>
                    {"".join(f"<li>{exp}</li>" for exp in processed_data.get('experience_highlights', []))}
                </ul>

                <h3>Next Steps:</h3>
                <p>{processed_data.get('next_steps', 'I will be reviewing your profile and matching you with relevant opportunities. You will receive an email from me when I find a great match for your skills and preferences.')}</p>

                <p>If you have any questions or would like to add anything to our conversation, feel free to reply to this email.</p>

                <p>Best regards,<br>
                <strong>Anita</strong><br>
                <em>Your AI Career Co-Pilot</em></p>
                """
            )

            logger.info("Creating SendGrid client...")
            sg = SendGridAPIClient(self.sendgrid_api_key)
            
            logger.info("Sending transcript summary email...")
            response = sg.send(message)
            
            logger.info(f"SendGrid response status code: {response.status_code}")
            logger.info(f"SendGrid response headers: {response.headers}")
            
            # Return a serializable response
            return {
                'status': 'success',
                'recipient': email,
                'subject': str(message.subject),  # Convert Subject to string
                'response_code': response.status_code
            }
        except Exception as e:
            logger.error(f"Error sending transcript summary email: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e),
                'recipient': email
            }