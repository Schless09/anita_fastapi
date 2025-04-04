from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
# import google.auth
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# import pickle
from app.config import get_settings, get_sendgrid_client  # Removed get_google_calendar_client
from pydantic import Field
from .base import parse_llm_json_response

class EmailTool(BaseTool):
    """Tool for handling email communications."""
    
    name = "email"
    description = "Handle email communications"
    # Define fields that will be set in __init__
    llm: ChatOpenAI = Field(default=None)
    settings: Any = Field(default=None) # Store settings
    sg_client: Optional[SendGridAPIClient] = Field(default=None) # Store SendGrid client
    from_email: Optional[str] = Field(default=None)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self):
        """Initialize the email tool."""
        super().__init__()
        
        # Get settings and store them
        self.settings = get_settings()
        
        # Set up LLM
        self.llm = ChatOpenAI(
            model_name=self.settings.openai_model, # Use model from settings
            temperature=0.7,
            api_key=self.settings.openai_api_key # Use key from settings
        )
        
        # Initialize SendGrid Client using the config function
        self.sg_client = get_sendgrid_client() # Returns None if not configured
        if self.sg_client:
            self.from_email = self.settings.sender_email
        else:
            self.from_email = None
            logger.warning("SendGrid client not initialized. Email sending disabled.")

    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Run email operations."""
        # Check if SendGrid is available for send operations
        if operation == "send_email" and not self.sg_client:
            return {
                "status": "error",
                "error": "SendGrid is not configured. Cannot send email."
            }
            
        try:
            if operation == "send_email":
                return self._send_email(**kwargs)
            elif operation == "generate_email":
                return self._generate_email(**kwargs)
            else:
                return {
                    "status": "error",
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Async version of email operations."""
        return self._run(operation, **kwargs)

    def _send_email(
        self,
        to_email: str,
        subject: str,
        content: str
    ) -> Dict[str, Any]:
        """Send an email using SendGrid."""
        if not self.sg_client or not self.from_email:
            return {
                "status": "error",
                "error": "SendGrid client or sender email is not configured."
            }
            
        try:
            message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                html_content=content
            )
            
            response = self.sg_client.send(message) # Use self.sg_client
            
            return {
                "status": "success",
                "success": response.status_code == 202,
                "message_id": response.headers.get("X-Message-Id")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_email(
        self,
        template: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate email content using LLM."""
        try:
            prompt = f"""Generate an email using this template and context:
            
            Template:
            {template}
            
            Context:
            {context}
            
            Return the email content in JSON format with subject and body fields."""
            
            response = self.llm.invoke(prompt)
            
            # Parse the response
            email_content = parse_llm_json_response(response.content)
            
            return {
                "status": "success",
                "subject": email_content.get("subject"),
                "body": email_content.get("body")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# class CalendarTool(BaseTool):
#     name = "calendar"
#     description = "Handle calendar operations"
#
#     def __init__(self):
#         super().__init__()
#         self.llm = ChatOpenAI(
#             model_name="gpt-4-turbo-preview",
#             temperature=0.3
#         )
#         self.creds = self._get_credentials()
#         self.service = build('calendar', 'v3', credentials=self.creds)
#
#     def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
#         """Run calendar operations."""
#         try:
#             if operation == "create_event":
#                 return self._create_event(**kwargs)
#             elif operation == "find_available_slots":
#                 return self._find_available_slots(**kwargs)
#             else:
#                 return {
#                     "status": "error",
#                     "error": f"Unknown operation: {operation}"
#                 }
#                 
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "error": str(e)
#             }
#
#     async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
#         """Async version of calendar operations."""
#         return self._run(operation, **kwargs)
#
#     def _get_credentials(self) -> Credentials:
#         """Get Google Calendar API credentials."""
#         creds = None
#         if os.path.exists('token.pickle'):
#             with open('token.pickle', 'rb') as token:
#                 creds = pickle.load(token)
#         
#         if not creds or not creds.valid:
#             if creds and creds.expired and creds.refresh_token:
#                 creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     'credentials.json', ['https://www.googleapis.com/auth/calendar']
#                 )
#                 creds = flow.run_local_server(port=0)
#             with open('token.pickle', 'wb') as token:
#                 pickle.dump(creds, token)
#         
#         return creds
#
#     def _create_event(
#         self,
#         summary: str,
#         description: str,
#         start_time: datetime,
#         end_time: datetime,
#         attendees: List[str]
#     ) -> Dict[str, Any]:
#         """Create a calendar event."""
#         try:
#             event = {
#                 'summary': summary,
#                 'description': description,
#                 'start': {
#                     'dateTime': start_time.isoformat(),
#                     'timeZone': 'UTC',
#                 },
#                 'end': {
#                     'dateTime': end_time.isoformat(),
#                     'timeZone': 'UTC',
#                 },
#                 'attendees': [{'email': email} for email in attendees],
#             }
#             
#             event = self.service.events().insert(
#                 calendarId='primary',
#                 body=event,
#                 sendUpdates='all'
#             ).execute()
#             
#             return {
#                 "status": "success",
#                 "event_id": event.get('id'),
#                 "html_link": event.get('htmlLink')
#             }
#             
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "error": str(e)
#             }
#
#     def _find_available_slots(
#         self,
#         start_time: datetime,
#         end_time: datetime,
#         duration_minutes: int = 60
#     ) -> Dict[str, Any]:
#         """Find available time slots in a given time range."""
#         try:
#             # Get busy times
#             body = {
#                 "timeMin": start_time.isoformat(),
#                 "timeMax": end_time.isoformat(),
#                 "items": [{"id": "primary"}]
#             }
#             
#             events_result = self.service.freebusy().query(body=body).execute()
#             busy_times = events_result.get('calendars', {}).get('primary', {}).get('busy', [])
#             
#             # Find available slots
#             available_slots = []
#             current_time = start_time
#             
#             while current_time < end_time:
#                 slot_end = current_time + timedelta(minutes=duration_minutes)
#                 
#                 # Check if slot overlaps with any busy times
#                 is_available = True
#                 for busy in busy_times:
#                     busy_start = datetime.fromisoformat(busy['start'].replace('Z', '+00:00'))
#                     busy_end = datetime.fromisoformat(busy['end'].replace('Z', '+00:00'))
#                     
#                     if (current_time < busy_end and slot_end > busy_start):
#                         is_available = False
#                         break
#                 
#                 if is_available:
#                     available_slots.append({
#                         "start": current_time.isoformat(),
#                         "end": slot_end.isoformat()
#                     })
#                 
#                 current_time += timedelta(minutes=30)
#             
#             return {
#                 "status": "success",
#                 "available_slots": available_slots
#             }
#             
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "error": str(e)
#             }
#
#     def find_available_slots(self, start_date: datetime, end_date: datetime, duration: timedelta) -> List[datetime]:
#         """Find available time slots in the calendar."""
#         service = get_google_calendar_client()
#         
#         # Get busy periods
#         body = {
#             "timeMin": start_date.isoformat() + "Z",
#             "timeMax": end_date.isoformat() + "Z",
#             "items": [{"id": "primary"}]
#         }
#         
#         events_result = service.freebusy().query(body=body).execute()
#         busy = events_result[0].get("busy", [])
#         
#         # Find free slots
#         current = start_date
#         available_slots = []
#         
#         while current + duration <= end_date:
#             is_free = True
#             for period in busy:
#                 busy_start = datetime.fromisoformat(period["start"].rstrip("Z"))
#                 busy_end = datetime.fromisoformat(period["end"].rstrip("Z"))
#                 
#                 if (current < busy_end and current + duration > busy_start):
#                     is_free = False
#                     current = busy_end
#                     break
#             
#             if is_free:
#                 available_slots.append(current)
#                 current += duration
#             else:
#                 current += timedelta(minutes=30)
#         
#         return available_slots 