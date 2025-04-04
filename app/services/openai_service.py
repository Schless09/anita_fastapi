from typing import Dict, Any, List
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import tempfile
import PyPDF2
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
        self.embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def extract_candidate_details(self, transcript: str) -> Dict[str, Any]:
        """
        Process transcript to extract candidate details using OpenAI.
        """
        try:
            prompt = f"""
            Extract the following information from this interview transcript:
            1. Current role and company
            2. Years of experience
            3. Tech stack and skills
            4. Previous companies and roles
            5. Education
            6. Career goals
            7. Salary expectations
            8. Work preferences (remote, hybrid, onsite)
            9. Location preferences
            10. Industry preferences

            Format the response as a JSON object with these fields:
            {{
                "current_role": "",
                "current_company": "",
                "years_of_experience": 0,
                "tech_stack": [],
                "previous_companies": [],
                "education": [],
                "career_goals": "",
                "salary_expectations": {{
                    "min": 0,
                    "max": 0,
                    "currency": "USD"
                }},
                "work_preferences": {{
                    "arrangement": [],
                    "location": []
                }},
                "industry_preferences": []
            }}

            Transcript:
            {transcript}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional recruiter extracting structured information from interview transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            raise Exception(f"Error extracting candidate details: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI's embedding model.
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def _prepare_text_for_embedding(self, data: Dict[str, Any]) -> str:
        """
        Convert structured data into text for embedding.
        """
        text_parts = []

        # Add basic information
        if data.get('current_role'):
            text_parts.append(f"Current role: {data['current_role']}")
        if data.get('current_company'):
            text_parts.append(f"Current company: {data['current_company']}")
        if data.get('years_of_experience'):
            text_parts.append(f"Years of experience: {data['years_of_experience']}")

        # Add tech stack
        if tech_stack := data.get('tech_stack', []):
            text_parts.append(f"Tech stack: {', '.join(tech_stack)}")

        # Add previous experience
        if prev_companies := data.get('previous_companies', []):
            text_parts.append(f"Previous companies: {', '.join(prev_companies)}")

        # Add education - handle both string list and dict list formats
        if education := data.get('education', []):
            if education and isinstance(education[0], dict):
                # Handle dictionary format
                edu_texts = []
                for edu in education:
                    parts = []
                    if degree := edu.get('degree'):
                        parts.append(degree)
                    if institution := edu.get('institution'):
                        parts.append(f"at {institution}")
                    if year := edu.get('year'):
                        parts.append(f"({year})")
                    edu_texts.append(" ".join(parts))
                text_parts.append(f"Education: {', '.join(edu_texts)}")
            else:
                # Handle string list format
                text_parts.append(f"Education: {', '.join(str(e) for e in education)}")

        # Add career goals
        if career_goals := data.get('career_goals', []):
            if isinstance(career_goals, list):
                text_parts.append(f"Career Goals: {', '.join(career_goals)}")
            else:
                text_parts.append(f"Career Goals: {career_goals}")

        # Add work preferences
        if work_prefs := data.get('work_preferences', {}):
            if isinstance(work_prefs, dict):
                prefs = []
                if arrangement := work_prefs.get('arrangement', []):
                    prefs.append(f"Work arrangement: {', '.join(arrangement)}")
                if location := work_prefs.get('location', []):
                    prefs.append(f"Location: {', '.join(location)}")
                if prefs:
                    text_parts.append(f"Work Preferences: {'; '.join(prefs)}")

        # Add industry preferences
        if industries := data.get('preferred_industries', []):
            text_parts.append(f"Industry Preferences: {', '.join(industries)}")

        return " | ".join(text_parts)

    def _parse_years_of_experience(self, years_str: str) -> float:
        """
        Parse years of experience from a string into a float.
        Handles cases like "Approximately 4", "4+", "4.5 years", etc.
        """
        try:
            # If it's already a number, return it
            if isinstance(years_str, (int, float)):
                return float(years_str)
            
            # Convert to string and clean it
            years_str = str(years_str).lower().strip()
            
            # Handle empty or zero cases
            if not years_str or years_str == '0':
                return 0.0
            
            # Remove common words and characters
            years_str = years_str.replace('approximately', '').replace('about', '')
            years_str = years_str.replace('years', '').replace('year', '')
            years_str = years_str.replace('of', '').replace('experience', '')
            years_str = years_str.strip()
            
            # Handle ranges (take the average)
            if '-' in years_str:
                parts = years_str.split('-')
                if len(parts) == 2:
                    try:
                        start = float(parts[0].strip().rstrip('+'))
                        end = float(parts[1].strip().rstrip('+'))
                        return (start + end) / 2
                    except ValueError:
                        pass
            
            # Handle "X+" format
            if years_str.endswith('+'):
                years_str = years_str.rstrip('+')
            
            # Try to convert to float
            return float(years_str)
            
        except (ValueError, TypeError):
            # If all parsing fails, return 0
            return 0.0

    async def extract_resume_information(self, resume_content: bytes) -> Dict[str, Any]:
        """
        Extract structured information from a resume PDF.
        """
        try:
            # Convert PDF bytes to text
            text = await self._extract_text_from_pdf(resume_content)
            
            # Create prompt for GPT
            prompt = f"""
            Extract the following information from this resume text. Return a JSON object with these fields:
            - current_role: Current or most recent job title
            - current_company: Current or most recent company
            - years_of_experience: Total years of professional experience (can be approximate)
            - tech_stack: Array of technologies and skills
            - previous_companies: Array of previous companies
            - education: Array of education entries
            - career_goals: Brief description of career objectives or goals
            - salary_expectations: Object with min and max values in USD (use 0 if not found)
            - work_preferences: Object with remote_preference (string), preferred_location (string), and willing_to_relocate (boolean)
            - industry_preferences: Array of preferred industries

            Resume text:
            {text}
            """
            
            # Call GPT to extract information
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{
                    "role": "system",
                    "content": "You are a professional resume parser. Extract structured information from resumes accurately. Use empty values or zeros when information is not found."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            extracted_info = json.loads(response.choices[0].message.content)
            
            # Get nested dictionaries with default values
            salary_expectations = extracted_info.get("salary_expectations", {})
            if not isinstance(salary_expectations, dict):
                salary_expectations = {}
                
            work_preferences = extracted_info.get("work_preferences", {})
            if not isinstance(work_preferences, dict):
                work_preferences = {}
            
            # Validate and clean the extracted information
            cleaned_info = {
                "current_role": str(extracted_info.get("current_role", "")),
                "current_company": str(extracted_info.get("current_company", "")),
                "years_of_experience": self._parse_years_of_experience(extracted_info.get("years_of_experience", 0)),
                "tech_stack": list(extracted_info.get("tech_stack", [])),
                "previous_companies": list(extracted_info.get("previous_companies", [])),
                "education": list(extracted_info.get("education", [])),
                "career_goals": str(extracted_info.get("career_goals", "")),
                "salary_expectations": {
                    "min": int(salary_expectations.get("min", 0) or 0),
                    "max": int(salary_expectations.get("max", 0) or 0)
                },
                "work_preferences": {
                    "remote_preference": str(work_preferences.get("remote_preference", "")),
                    "preferred_location": str(work_preferences.get("preferred_location", "")),
                    "willing_to_relocate": bool(work_preferences.get("willing_to_relocate", False))
                },
                "industry_preferences": list(extracted_info.get("industry_preferences", []))
            }
            
            return cleaned_info
            
        except Exception as e:
            logger.error(f"Error extracting resume information: {str(e)}")
            raise

    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content.
        """
        try:
            # Create a temporary file to write PDF content
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_path = temp_file.name

            try:
                # Use PyPDF2 to extract text
                text = ""
                with open(temp_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                return text.strip()
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise 

    async def quick_extract_current_position(self, text: str) -> Dict[str, str]:
        """
        Quickly extract current role and company from resume text.
        This is a fast extraction for the initial Retell call.
        """
        try:
            prompt = f"""
            From this resume text, extract ONLY the current or most recent:
            1. Job title/role
            2. Company name

            Return as a JSON object with these fields:
            - current_role: The job title/role
            - current_company: The company name

            If you can't find the information, use "current role" and "current company" as defaults.
            Be brief and quick - we only need these two pieces of information.

            Resume text:
            {text}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{
                    "role": "system",
                    "content": "You are a quick resume scanner. Extract only the current position information."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                response_format={ "type": "json_object" },
                max_tokens=100  # Keep response short for speed
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "current_role": str(result.get("current_role", "current role")),
                "current_company": str(result.get("current_company", "current company"))
            }
            
        except Exception as e:
            logger.error(f"Error in quick position extraction: {str(e)}")
            return {
                "current_role": "current role",
                "current_company": "current company"
            }

    async def extract_transcript_info(self, transcript: str) -> Dict[str, Any]:
        """
        Extract structured information from call transcript using OpenAI.
        Returns a dictionary with the extracted information.
        Handles partial conversations and captures any available information.
        """
        try:
            system_message = """You are an expert at extracting structured information from call transcripts.
            Extract ANY information that is available, even from partial conversations.
            Focus on capturing:
            1. Any explicitly stated preferences (work arrangements, benefits, company size, etc.)
            2. Any mentioned experience or skills
            3. Any stated career goals or preferences
            4. Any mentioned education or qualifications
            5. Any industry preferences or restrictions
            6. Any technical preferences or restrictions
            
            Do NOT include any personal information like name, email, phone, or location.
            Keep the information professional and focused on qualifications and preferences.
            
            If a field is not mentioned in the conversation, return an empty value for that field.
            Do not make assumptions about missing information."""
            
            user_message = f"""Please extract any available information from this call transcript:
            {transcript}
            
            Return the information in this exact JSON structure, with empty values for any fields not mentioned:
            {{
                "professional_summary": "string or empty string",
                "current_role": "string or empty string",
                "current_company": "string or empty string",
                "years_of_experience": "string or empty string",
                "tech_stack": ["string"],
                "skills": ["string"],
                "experience": [
                    {{
                        "title": "string",
                        "company": "string",
                        "duration": "string",
                        "description": "string"
                    }}
                ],
                "education": [
                    {{
                        "degree": "string",
                        "institution": "string",
                        "year": "string"
                    }}
                ],
                "career_goals": ["string"],
                "work_preferences": {{
                    "arrangement": ["string"],
                    "benefits": ["string"],
                    "company_size": ["string"],
                    "interview_preference": "string"
                }},
                "preferred_industries": ["string"],
                "preferred_project_types": ["string"],
                "project_visibility_preference": ["string"],
                "desired_company_stage": ["string"],
                "preferred_company_size": ["string"],
                "technologies_to_avoid": ["string"],
                "call_status": {{
                    "is_complete": false,
                    "reason": "string",
                    "follow_up_needed": true
                }}
            }}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )
            
            if response.choices and response.choices[0].message.content:
                extracted_info = json.loads(response.choices[0].message.content)
                
                # Analyze if the call was complete
                transcript_lines = transcript.split('\n')
                if len(transcript_lines) < 20:  # Arbitrary threshold for very short conversations
                    extracted_info['call_status'] = {
                        'is_complete': False,
                        'reason': 'Conversation too short',
                        'follow_up_needed': True
                    }
                elif not any('experience' in line.lower() or 'background' in line.lower() for line in transcript_lines):
                    extracted_info['call_status'] = {
                        'is_complete': False,
                        'reason': 'No background/experience discussed',
                        'follow_up_needed': True
                    }
                else:
                    extracted_info['call_status'] = {
                        'is_complete': True,
                        'reason': 'Sufficient information gathered',
                        'follow_up_needed': False
                    }
                
                logger.info(f"Successfully extracted transcript info: {json.dumps(extracted_info, indent=2)}")
                return extracted_info
            else:
                logger.error("No content in OpenAI response")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting transcript info: {str(e)}")
            return {} 