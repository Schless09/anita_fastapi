import os
import logging
import ssl
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, From, Subject, HtmlContent, PlainTextContent
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.sendgrid import SENDGRID_CONFIG
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import uuid
import json
from openai import OpenAI
from pinecone import Pinecone
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.slack import send_slack_notification

# Set up logging
logger = logging.getLogger(__name__)

class InteractionAgent:
    def __init__(self, pinecone_client: Optional[Pinecone] = None):
        """Initialize the InteractionAgent with optional Pinecone client."""
        self.pc = pinecone_client
        self.openai = OpenAI()
        
        # Initialize SendGrid configuration
        self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
        self.sender_email = os.getenv('SENDER_EMAIL')
        
        # Initialize Pinecone if client provided
        if self.pc:
            try:
                # Use the call-statuses index for conversation history
                self.conversations_index = self.pc.Index(os.getenv('PINECONE_CALL_STATUS_INDEX', 'call-statuses'))
                logger.info(f"Using call status index: {os.getenv('PINECONE_CALL_STATUS_INDEX', 'call-statuses')}")
            except Exception as e:
                logger.error(f"❌ Error initializing Pinecone: {str(e)}")
                self.conversations_index = None
        else:
            self.conversations_index = None
        
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
                <strong>Company:</strong> <a href="{job_details.get('company_url', '#')}" target="_blank">{company}</a><br>

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

    async def send_missed_call_email(self, email: str, first_name: str, candidate_id: str):
        """Send a 'sorry I missed you' email to candidates who missed their calls."""
        try:
            # Get base URL from environment variable, default to ngrok for development
            base_url = os.getenv('BASE_URL', 'https://85ac-2601-645-8000-fe70-4981-3c72-e8ef-490f.ngrok-free.app')
            
            # Create call-back URL with candidate_id
            call_back_url = f"{base_url}/calls/initiate/{candidate_id}"
            
            # Create the email message using SendGrid's Mail class
            message = Mail(
                from_email=self.sender_email,
                to_emails=email,
                subject="Sorry I Missed You - Let's Reschedule",
                html_content=f"""
                <html>
                    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                            <h2 style="color: #2c3e50;">Hi {first_name or 'there'},</h2>
                            <p>I apologize for missing our call. I'd love to reschedule and continue our conversation about your career opportunities.</p>
                            <p>You can easily schedule a call back by clicking the button below:</p>
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="{call_back_url}" 
                                   style="background-color: #3498db; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">
                                    Schedule a Call Back
                                </a>
                            </div>
                            <p>Or if you prefer, you can simply reply to this email and I'll get back to you to schedule a better time.</p>
                            <p>Looking forward to our conversation!</p>
                            <p>Best regards,<br>Anita</p>
                        </div>
                    </body>
                </html>
                """
            )

            # Send the email
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            logger.info(f"✅ Successfully sent missed call email to {email}")
            
            return {
                'status': 'success',
                'recipient': email,
                'subject': message.subject,
                'response_code': response.status_code
            }
        except Exception as e:
            logger.error(f"❌ Error sending missed call email: {str(e)}")
            raise

    async def send_transcript_summary(self, email: str, processed_data: Dict[str, Any], call_status: str = "completed") -> Dict[str, Any]:
        """Send a summary email to a candidate with their transcript analysis."""
        try:
            # Handle different call scenarios
            if call_status == "missed" or call_status == "short":
                return await self.send_missed_call_email(email, processed_data.get('first_name', ''), processed_data.get('candidate_id', ''))
            
            logger.info(f"Preparing transcript summary email for {email}")
            
            # Get first name, defaulting to empty string if not found
            first_name = processed_data.get('first_name', '')
            logger.info(f"Using first name: {first_name}")
            
            # Check if we have sufficient key points data
            key_points = processed_data.get('key_points', [])
            has_sufficient_data = len(key_points) > 0
            
            # Create the email message
            message = Mail(
                from_email=Email(self.sender_email),
                to_emails=To(email),
                subject="Thanks for speaking with me!",
                html_content=f"""
                <p>Hi{', ' + first_name if first_name else ' there'},</p>
                
                <p>Thank you for taking the time to speak with me today!</p>

                {
                    f'''
                    <h3>Key Points from Our Conversation:</h3>
                    <ul>
                        {"".join(f"<li>{point}</li>" for point in key_points)}
                    </ul>
                    '''
                    if has_sufficient_data else
                    '''
                    <p>I didn't catch much from our conversation today. Let's have another call to better understand your background and preferences! Let me know what time works best for you.</p>
                    '''
                }

                <p>{processed_data.get('next_steps', 'I will be reviewing your profile and matching you with relevant opportunities. You will receive an email from me when I find a great match for your skills and preferences.')}</p>

                <p>If you have any questions or would like to add anything to our conversation, feel free to reply to this email. The more information you provide, the better I can match you with the right opportunities.</p>

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
            
            return {
                'status': 'success',
                'recipient': email,
                'subject': str(message.subject),
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

    async def send_email(self, to_email: str, subject: str, content: str) -> Dict[str, Any]:
        """Send a general email to a recipient."""
        try:
            logger.info(f"📧 Sending email to {to_email}")
            logger.info(f"Subject: {subject}")
            
            message = Mail(
                from_email=self.sender_email,
                to_emails=to_email,
                subject=subject,
                html_content=content
            )
            
            logger.info("Creating SendGrid client...")
            sg = SendGridAPIClient(self.sendgrid_api_key)
            
            logger.info("Sending email...")
            response = sg.send(message)
            
            logger.info(f"SendGrid response status code: {response.status_code}")
            logger.info(f"SendGrid response headers: {response.headers}")
            
            return {
                'status': 'success',
                'recipient': to_email,
                'subject': subject,
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
                'recipient': to_email
            }

    async def store_conversation(self, candidate_id: str, email_data: Dict[str, Any], is_incoming: bool = True) -> str:
        """Store a conversation entry in Pinecone with proper context."""
        try:
            # Create conversation entry
            conversation_id = f"conv_{uuid.uuid4()}"
            timestamp = datetime.utcnow().isoformat()
            
            # Generate embedding for the email content
            content = email_data.get('text', '') or email_data.get('html', '')
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )
            embedding = response.data[0].embedding
            
            # Prepare metadata
            metadata = {
                'candidate_id': candidate_id,
                'timestamp': timestamp,
                'subject': email_data.get('subject', ''),
                'direction': 'incoming' if is_incoming else 'outgoing',
                'conversation_id': conversation_id,
                'email_from': email_data.get('from', ''),
                'email_to': email_data.get('to', ''),
                'content_preview': content[:100]  # Store preview for quick access
            }
            
            # Store in Pinecone
            self.conversations_index.upsert(
                vectors=[{
                    'id': conversation_id,
                    'values': embedding,
                    'metadata': metadata
                }]
            )
            
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            raise

    async def get_conversation_history(self, candidate_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent conversation history for a candidate."""
        try:
            # Query Pinecone for recent conversations
            results = self.conversations_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata filtering
                filter={'candidate_id': candidate_id},
                top_k=limit,
                include_metadata=True
            )
            
            # Sort conversations by timestamp
            conversations = sorted(
                [match.metadata for match in results.matches],
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []

    async def analyze_email_intent(self, content: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze email content for intent and potential security issues."""
        try:
            # Prepare conversation context
            context = "\n".join([
                f"Previous {msg['direction']} message ({msg['timestamp']}):\n{msg['content_preview']}"
                for msg in conversation_history
            ])
            
            # Analyze with OpenAI
            response = self.openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": """Analyze this email in the context of previous conversations.
                    Check for:
                    1. Intent (question, feedback, application, etc.)
                    2. Security concerns (suspicious patterns, potential system gaming)
                    3. Sentiment
                    4. Required action
                    5. Job relevance
                    
                    Format response as JSON with these keys:
                    {
                        "intent": str,
                        "security_concerns": List[str],
                        "sentiment": str,
                        "required_action": str,
                        "job_relevant": bool,
                        "needs_human_review": bool
                    }"""},
                    {"role": "user", "content": f"Previous conversations:\n{context}\n\nCurrent email:\n{content}"}
                ],
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing email intent: {str(e)}")
            return {
                "intent": "unknown",
                "security_concerns": [],
                "sentiment": "neutral",
                "required_action": "human_review",
                "job_relevant": False,
                "needs_human_review": True
            }

    async def process_candidate_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming candidate email with enhanced security and context."""
        try:
            # Extract candidate email
            candidate_email = email_data['from']
            
            # Get conversation history
            conversation_history = await self.get_conversation_history(candidate_email)
            
            # Store the new conversation entry
            conversation_id = await self.store_conversation(candidate_email, email_data)
            
            # Analyze email intent and security
            analysis = await self.analyze_email_intent(email_data['text'], conversation_history)
            
            # Check for security concerns
            if analysis['security_concerns']:
                await send_slack_notification(
                    f"*🚨 Security Concern in Candidate Email*\n"
                    f"Candidate: {candidate_email}\n"
                    f"Concerns: {', '.join(analysis['security_concerns'])}"
                )
                return {
                    'status': 'error',
                    'message': 'Security concerns detected',
                    'needs_human_review': True
                }
            
            # Process based on intent and context
            if analysis['needs_human_review']:
                await send_slack_notification(
                    f"*👥 Human Review Needed*\n"
                    f"Candidate: {candidate_email}\n"
                    f"Intent: {analysis['intent']}\n"
                    f"Sentiment: {analysis['sentiment']}\n"
                    f"Action Needed: {analysis['required_action']}"
                )
                return {
                    'status': 'success',
                    'message': 'Forwarded for human review',
                    'needs_human_review': True
                }
            
            # Generate and send appropriate response
            response_content = await self.generate_response(
                email_data,
                analysis,
                conversation_history
            )
            
            await self.send_email(
                to_email=candidate_email,
                subject=f"Re: {email_data['subject']}",
                content=response_content
            )
            
            return {
                'status': 'success',
                'conversation_id': conversation_id,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error processing candidate email: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def generate_response(
        self,
        email_data: Dict[str, Any],
        analysis: Dict[str, Any],
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Generate an appropriate response based on email analysis and job matches."""
        try:
            # Get candidate's matched jobs
            candidate_email = email_data['from']
            matched_jobs = await self.get_candidate_matched_jobs(candidate_email)
            
            # Prepare conversation context
            context = "\n".join([
                f"Previous {msg['direction']} message ({msg['timestamp']}):\n{msg['content_preview']}"
                for msg in conversation_history[-3:]  # Last 3 messages
            ])
            
            # Prepare job context (only matched jobs)
            job_context = []
            for job in matched_jobs:
                job_context.append(f"""
                Job ID: {job['id']}
                Title: {job['title']}
                Company: {job['company_name']}
                Status: {job['status']}
                Match Score: {job['match_score']}
                """)
            
            # Generate response with OpenAI
            response = self.openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": """You are an AI recruiter assistant. Follow these rules strictly:
                    1. Only discuss jobs that are in the provided matched_jobs list
                    2. Never mention or hint at other opportunities
                    3. Be professional but friendly
                    4. If asked about jobs not in the matched list, politely explain you can only discuss current matches
                    5. Keep responses focused and relevant to the candidate's questions
                    6. Include clear next steps or calls to action
                    7. Maintain conversation context
                    8. Use appropriate tone based on sentiment analysis"""},
                    {"role": "user", "content": f"""
                    Candidate Email: {email_data['text']}
                    
                    Analysis:
                    Intent: {analysis['intent']}
                    Sentiment: {analysis['sentiment']}
                    Job Relevant: {analysis['job_relevant']}
                    
                    Previous Conversation:
                    {context}
                    
                    Matched Jobs:
                    {"".join(job_context) if job_context else "No current job matches"}
                    
                    Generate an appropriate response:"""}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            response_content = response.choices[0].message.content
            
            # Store outgoing message in conversation history
            await self.store_conversation(
                candidate_email,
                {
                    'subject': f"Re: {email_data['subject']}",
                    'text': response_content,
                    'from': self.sender_email,
                    'to': candidate_email
                },
                is_incoming=False
            )
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return """I apologize, but I'm having trouble generating a response at the moment. 
                     A human recruiter will review your message and get back to you soon."""
    
    async def get_candidate_matched_jobs(self, candidate_email: str) -> List[Dict[str, Any]]:
        """Retrieve only the jobs that have been matched with the candidate."""
        try:
            # Query Pinecone for candidate's matched jobs
            results = self.conversations_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata filtering
                filter={
                    'candidate_email': candidate_email,
                    'status': {'$in': ['MATCHED', 'INTERVIEWING', 'OFFERED']}
                },
                top_k=10,
                include_metadata=True
            )
            
            # Extract job details
            matched_jobs = []
            for match in results.matches:
                job_data = match.metadata
                # Only include essential job information
                matched_jobs.append({
                    'id': job_data['job_id'],
                    'title': job_data['job_title'],
                    'company_name': job_data['company_name'],
                    'status': job_data['status'],
                    'match_score': match.score
                })
            
            return matched_jobs
            
        except Exception as e:
            logger.error(f"Error retrieving matched jobs: {str(e)}")
            return []