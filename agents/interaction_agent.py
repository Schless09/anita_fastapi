import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Header
from dotenv import load_dotenv
import sendgrid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import re
from .vector_store import VectorStore

print("SendGrid library is installed and accessible.")

class InteractionAgent:
    def __init__(self):
        load_dotenv()
        self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.human_support_email = "andrew@thecodercollective.com"
        self.candidate_profiles = {}
        self.vector_store = None  # Will be initialized later if needed
        self.conversation_history = {}  # Store conversation history by candidate_id

    def update_candidate_knowledge(self, candidate_id: str, profile_data: Dict[str, Any]):
        """Update our knowledge about a candidate based on their profile data."""
        self.candidate_profiles[candidate_id] = profile_data

    def get_personalized_greeting(self, candidate_id: str) -> str:
        """Generate a personalized greeting based on candidate data."""
        profile = self.candidate_profiles.get(candidate_id, {})
        candidate_name = profile.get('processed_data', {}).get('candidate_name', '')
        current_role = profile.get('processed_data', {}).get('current_role', '')
        
        if candidate_name:
            greeting = f"Hi {candidate_name.split()[0]}"
            if current_role:
                greeting += f", hope you're doing well in your role as {current_role}"
            return greeting + "!"
        
        return "Hi there!"

    def evaluate_job_match(self, candidate_id: str, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how well a job matches the candidate's preferences."""
        profile = self.candidate_profiles.get(candidate_id, {})
        processed_data = profile.get('processed_data', {})
        
        match_reasons = []
        dealbreakers = []
        
        # Location match
        if processed_data.get('preferred_locations'):
            candidate_cities = [loc.get('city', '').lower() for loc in processed_data['preferred_locations']]
            job_cities = [city.lower() for city in job_details.get('role_details', {}).get('city', [])]
            
            if any(city in candidate_cities for city in job_cities):
                match_reasons.append("Location matches preferences")
        
        # Salary match
        if processed_data.get('salary_expectations', {}).get('min'):
            job_salary = job_details.get('salary_range', {})
            if job_salary.get('max', 0) >= processed_data['salary_expectations']['min']:
                match_reasons.append("Salary meets expectations")
            else:
                dealbreakers.append("Salary below minimum expectations")
        
        # Tech stack match
        if processed_data.get('tech_stack') and job_details.get('tech_stack'):
            matching_tech = set(job_details['tech_stack']) & set(processed_data['tech_stack'])
            if matching_tech:
                match_reasons.append(f"Matching technologies: {', '.join(matching_tech)}")
        
        return {
            "match_quality": "poor" if dealbreakers else "good" if match_reasons else "neutral",
            "match_reasons": match_reasons,
            "dealbreakers": dealbreakers
        }

    def generate_personalized_content(self, candidate_id: str, job_details: Dict[str, Any]) -> str:
        """Generate personalized email content based on candidate preferences and job details."""
        profile = self.candidate_profiles.get(candidate_id, {})
        processed_data = profile.get('processed_data', {})
        
        # Start with HTML template
        content_parts = ['<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">']
        
        # Personalized greeting
        content_parts.append(f'<p>{self.get_personalized_greeting(candidate_id)}</p>')
        
        # Job introduction
        content_parts.append('<p>I came across an exciting opportunity that aligns with your interests and experience.</p>')
        
        # Role details in a styled box
        content_parts.append('''
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
                <p style="margin: 0;"><strong>Role:</strong> {title}</p>
                <p style="margin: 5px 0;"><strong>Company:</strong> {company}</p>
            </div>
        '''.format(title=job_details.get('title'), company=job_details.get('company')))
        
        # Personalized match points
        match_eval = self.evaluate_job_match(candidate_id, job_details)
        if match_eval['match_reasons']:
            content_parts.append('<p>I thought this would be particularly interesting for you because:</p>')
            content_parts.append('<ul style="margin: 10px 0; padding-left: 20px;">')
            for reason in match_eval['match_reasons']:
                content_parts.append(f'<li>{reason}</li>')
            content_parts.append('</ul>')
        
        # Tech stack alignment if relevant
        if processed_data.get('tech_stack') and job_details.get('tech_stack'):
            matching_tech = set(job_details['tech_stack']) & set(processed_data['tech_stack'])
            if matching_tech:
                content_parts.append(f'<p>The role involves technologies you\'re experienced with: <strong>{", ".join(matching_tech)}</strong></p>')
        
        # Career growth alignment
        if processed_data.get('career_goals'):
            content_parts.append(f'<p>This opportunity could align well with your career goals in {", ".join(processed_data["career_goals"])}.</p>')
        
        # Call to action
        content_parts.append('''
            <p style="margin-top: 20px;">Would you be interested in learning more about this role? If so, I'd be happy to provide additional details and discuss next steps.</p>
        ''')
        
        # Closing
        content_parts.append('''
            <p style="margin-top: 20px;">Looking forward to hearing from you!</p>
            <p style="margin-top: 20px;">
                Yours in Recruitment,<br>
                <span style="color: #2b5a8e; font-weight: bold;">Anita</span>
            </p>
        ''')
        
        # Close the main div
        content_parts.append('</div>')
        
        return '\n'.join(content_parts)

    def add_thread_headers(self, message: Mail, job_id: str, candidate_id: str, thread_type: str = 'reply'):
        """Add thread tracking headers to a SendGrid message."""
        message.header = Header("X-Job-ID", job_id)
        message.header = Header("X-Candidate-ID", candidate_id)
        message.header = Header("X-Thread-Type", thread_type)
        return message

    def contact_candidate(self, candidate_id: str, job_match: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a personalized email to the candidate about the job opportunity."""
        # Store job data in conversation history
        if 'id' in job_match:
            self.conversation_history[job_match['id']] = job_match
        
        profile = self.candidate_profiles.get(candidate_id)
        if not profile:
            return {
                'status': 'error',
                'error': 'Candidate profile not found',
                'candidate_id': candidate_id
            }

        # Get candidate email from processed data
        processed_data = profile.get('processed_data', {})
        recipient_email = processed_data.get('contact_information', '')
        if not recipient_email:
            return {
                'status': 'error',
                'error': 'Candidate email not found',
                'candidate_id': candidate_id
            }

        # Generate personalized content
        email_content = self.generate_personalized_content(candidate_id, job_match)

        # Add thread identifiers in a hidden div
        thread_info = f'''
        <div style="display:none">
            candidate_id:{candidate_id}
            job_id:{job_match['id']}
        </div>
        '''
        email_content = email_content.replace('</div>', f'{thread_info}</div>')

        # Create the email message with custom headers
        message = Mail(
            from_email=self.sender_email,
            to_emails=recipient_email,
            subject=f"Exciting Opportunity: {job_match.get('title')} at {job_match.get('company')} [job_id={job_match['id']};candidate_id={candidate_id}]",
            html_content=email_content
        )
        
        # Add thread tracking headers
        message = self.add_thread_headers(message, job_match['id'], candidate_id, 'initial_contact')

        try:
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            
            return {
                'status': 'success',
                'recipient': recipient_email,
                'subject': message.subject,
                'response_code': response.status_code,
                'personalization_data': {
                    'match_evaluation': self.evaluate_job_match(candidate_id, job_match),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'recipient': recipient_email,
                'candidate_id': candidate_id
            }

    def extract_questions(self, email_content: str) -> List[str]:
        """Extract questions from email content using various patterns."""
        questions = []
        
        # Pattern 1: Sentences ending with question mark
        question_pattern1 = r'[^.!?]*\?'
        questions.extend(re.findall(question_pattern1, email_content))
        
        # Pattern 2: Common question starters
        question_starters = r'(can|could|what|when|where|which|who|why|how|is|are|will|would|should|do|does|did).*?[.!?]'
        potential_questions = re.findall(question_starters, email_content, re.IGNORECASE)
        questions.extend([q for q in potential_questions if '?' in q])
        
        return list(set(questions))

    def search_job_information(self, job_id: str, question: str) -> Optional[Dict[str, Any]]:
        """Search for job-related information in the vector database or direct job data."""
        try:
            # Create a mapping of question topics to job data fields
            topic_mapping = {
                r'salary|compensation|pay': ('role_details', 'salary_range'),
                r'location|where|office': ('role_details', 'city'),
                r'tech|stack|technology|framework': ('tech_stack',),
                r'company|about': ('description',),
                r'requirements|qualifications': ('role_details', 'requirements'),
                r'benefits|perks': ('role_details', 'benefits'),
                r'remote|work from home': ('role_details', 'work_arrangement'),
                r'interview|process': ('role_details', 'interview_process'),
                r'team|group|department': ('role_details', 'team'),
                r'responsibilities|duties': ('role_details', 'responsibilities')
            }
            
            # Get job data from vector store or direct data
            if self.vector_store is not None:
                job_response = self.vector_store.jobs_index.fetch(ids=[job_id])
                if not job_response.vectors:
                    return None
                job_data = job_response.vectors[job_id].metadata
            else:
                # Use the job data directly from the conversation history
                job_data = self.conversation_history.get(job_id)
                if not job_data:
                    return None
            
            # Find relevant information based on question
            for pattern, fields in topic_mapping.items():
                if re.search(pattern, question, re.IGNORECASE):
                    # Navigate through nested dictionary using the fields tuple
                    current_data = job_data
                    for field in fields:
                        if isinstance(current_data, dict) and field in current_data:
                            current_data = current_data[field]
                        else:
                            current_data = None
                            break
                    
                    if current_data:
                        return {
                            'topic': pattern.split('|')[0],
                            'information': current_data
                        }
            
            return None
            
        except Exception as e:
            print(f"Error searching job information: {str(e)}")
            return None

    def format_answer(self, question: str, answer_data: Dict[str, Any]) -> str:
        """Format the answer in a natural, conversational way."""
        topic = answer_data['topic']
        info = answer_data['information']
        
        # Format based on topic
        if topic == 'salary':
            return f"Regarding compensation, this role offers {info}."
        elif topic == 'location':
            if isinstance(info, list):
                locations = ', '.join(info)
                return f"The role is located in {locations}."
            return f"The role is located in {info}."
        elif topic == 'tech':
            if isinstance(info, list):
                techs = ', '.join(info)
                return f"The technology stack for this role includes: {techs}."
            return f"The technology stack for this role includes: {info}."
        else:
            return str(info)

    def forward_to_human(self, candidate_id: str, email_content: str, job_id: str) -> Dict[str, Any]:
        """Forward the email to human support."""
        try:
            profile = self.candidate_profiles.get(candidate_id, {})
            candidate_name = profile.get('processed_data', {}).get('candidate_name', 'Unknown Candidate')
            
            forward_content = f'''
            <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <p><strong>HUMAN IN THE LOOP REQUIRED</strong></p>
                <p>Candidate: {candidate_name} (ID: {candidate_id})</p>
                <p>Job ID: {job_id}</p>
                <p>Original Email:</p>
                <div style="margin-left: 20px; padding: 10px; border-left: 2px solid #ccc;">
                    {email_content}
                </div>
                <p>Unable to automatically process this inquiry. Human response needed.</p>
            </div>
            '''
            
            message = Mail(
                from_email=self.sender_email,
                to_emails=self.human_support_email,
                subject=f"HUMAN IN THE LOOP - Question from {candidate_name}",
                html_content=forward_content
            )
            
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            
            return {
                'status': 'forwarded',
                'message': 'Email forwarded to human support',
                'response_code': response.status_code
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def handle_candidate_reply(self, candidate_id: str, email_content: str, job_id: str) -> Dict[str, Any]:
        """Process a candidate's email reply and generate appropriate response."""
        try:
            # Extract questions from the email
            questions = self.extract_questions(email_content)
            
            if not questions:
                return self.forward_to_human(candidate_id, email_content, job_id)
            
            # Try to answer each question
            answers = []
            needs_human = False
            
            for question in questions:
                answer_data = self.search_job_information(job_id, question)
                if answer_data:
                    answers.append(self.format_answer(question, answer_data))
                else:
                    needs_human = True
                    break
            
            if needs_human or not answers:
                return self.forward_to_human(candidate_id, email_content, job_id)
            
            # Send response to candidate
            response_content = f'''
            <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <p>Thank you for your questions! Here are the details you requested:</p>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    {"".join(f"<li>{answer}</li>" for answer in answers)}
                </ul>
                <p>Is there anything else you'd like to know about this opportunity?</p>
                <p style="margin-top: 20px;">
                    Yours in Recruitment,<br>
                    <span style="color: #2b5a8e; font-weight: bold;">Anita</span>
                </p>
                <div style="display:none">
                    candidate_id:{candidate_id}
                    job_id:{job_id}
                </div>
            </div>
            '''
            
            profile = self.candidate_profiles.get(candidate_id, {})
            recipient_email = profile.get('processed_data', {}).get('contact_information', '')
            
            message = Mail(
                from_email=self.sender_email,
                to_emails=recipient_email,
                subject=f"RE: Your questions about the position [job_id={job_id};candidate_id={candidate_id}]",
                html_content=response_content
            )
            
            # Add thread tracking headers
            message = self.add_thread_headers(message, job_id, candidate_id, 'reply')
            
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            
            return {
                'status': 'success',
                'message': 'Response sent to candidate',
                'answers': answers,
                'response_code': response.status_code
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }