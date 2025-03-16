import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

class InteractionAgent:
    def __init__(self):
        load_dotenv()
        self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
        self.sender_email = os.getenv('SENDER_EMAIL')  # Use the sender email from the environment

    def contact_candidate(self, job_match):
        """Sends an email to the candidate about the job opportunity."""
        recipient_email = job_match.get('email', 'harrrisonfranke@gmail.com')
        job_title = job_match.get('title', 'Senior Backend Engineer')
        company = job_match.get('company', 'Hedra')

        # Create the email message
        message = Mail(
            from_email=self.sender_email,  # Use the sender email from the environment
            to_emails=recipient_email,
            subject=f"Exciting Opportunity: {job_title} at {company}",
            html_content=f"""
            <p>Hi Andrew,</p>
            <p>I hope you're doing well! I came across an exciting opportunity that aligns with your profile and wanted to share it with you.</p>

            <p><strong>Role:</strong> {job_title}<br>
            <strong>Company:</strong> {company}</p>

            <p>You can find more details about the position here:<br>
            <a href="https://www.paraform.com/share/hedra/cm2pcqjin002il90czttxt1sy" target="_blank">Job Details</a></p>

            <p>Would you be interested in learning more? I'd be happy to hop on a quick call to discuss the role in detail and answer any questions you may have.</p>

            <p>If you'd like us to present your profile to the hiring manager, just reply with <strong>1-2 bullet points</strong> highlighting why you'd be a great fit, and I'll submit your application.</p>

            <p>Not interested? No worries! Let me know (along with <strong>1-2 reasons why</strong>)—this helps us refine our matching algorithm and ensure better opportunities for you in the future.</p>

            <p>Looking forward to your thoughts!</p>

            <p>Best,<br>
            <strong>Anita</strong><br>
            <em>Your AI Recruiter</em></p>
            """
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