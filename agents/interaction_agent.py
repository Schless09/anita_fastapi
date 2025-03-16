import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
import sendgrid

print("SendGrid library is installed and accessible.")

class InteractionAgent:
    def __init__(self):
        load_dotenv()
        self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
        self.sender_email = os.getenv('SENDER_EMAIL')  # Use the sender email from the environment

    def contact_candidate(self, job_match):
        """Sends an email to the candidate about the job opportunity."""
        recipient_email = "arschuessler90@gmail.com"
        job_title = job_match.get('title', 'Senior Backend Engineer')
        company = job_match.get('company', 'Hedra')

        # Create the email message
        message = Mail(
            from_email=self.sender_email,  # Use the sender email from the environment
            to_emails=recipient_email,
            subject=f"Exciting Opportunity: {job_title} at {company}",
            html_content=f"""
            <p>Hi Andrew,</p>
            <p>I hope this email finds you well! I wanted to reach out because I found an exciting opportunity that matches your profile.</p>
            <p>Role: {job_title}<br>Company: {company}</p>
            <p>You can learn more about this position here:<br>
            <a href="https://www.paraform.com/share/hedra/cm2pcqjin002il90czttxt1sy">Job Details</a></p>
            <p>Would you be interested in learning more about this role? If so, I'd be happy to provide additional details and discuss next steps.</p>
            <p>Looking forward to hearing from you!</p>
            <p>Best regards,<br>Your AI Recruiter</p>
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