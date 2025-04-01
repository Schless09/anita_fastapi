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
        Extract structured information from a call transcript using OpenAI.
        """
        try:
            # Create a system message that defines the structure we want
            system_message = """You are an expert at analyzing job candidate interviews and extracting structured information.
            Extract the following information from the transcript and return it in a structured JSON format.
            If information is not found, use empty values (empty strings, empty lists, 0, or false) as appropriate.
            
            Required fields in the JSON response:
            - previous_companies: List of companies the candidate has worked at
            - tech_stack: List of technologies and skills mentioned
            - years_of_experience: Number of years of experience (as a number)
            - industries: List of industries mentioned
            - undesired_industries: List of industries the candidate wants to avoid
            - company_size_at_join: Size of company when candidate joined (as a number)
            - current_company_size: Current size of company (as a number)
            - company_stage: Current stage of company (e.g., startup, enterprise)
            - experience_with_significant_company_growth: Boolean indicating if they've experienced company growth
            - early_stage_startup_experience: Boolean indicating if they've worked at early stage startups
            - leadership_experience: Boolean indicating if they have leadership experience
            - preferred_work_arrangement: List of preferred work arrangements (e.g., remote, hybrid)
            - preferred_locations: List of preferred locations
            - visa_sponsorship_needed: Boolean indicating if they need visa sponsorship
            - salary_expectations: Object with min and max values
            - desired_company_stage: List of preferred company stages
            - preferred_industries: List of preferred industries
            - preferred_product_types: List of preferred product types
            - motivation_for_job_change: List of reasons for job change
            - work_life_balance_preferences: String describing work-life balance preferences
            - desired_company_culture: String describing desired company culture
            - traits_to_avoid_detected: List of traits to avoid
            - additional_notes: String with any additional notes
            - candidate_tags: List of relevant tags
            - next_steps: String describing next steps
            - role_preferences: List of preferred roles
            - technologies_to_avoid: List of technologies to avoid
            - company_culture_preferences: List of company culture preferences
            - work_environment_preferences: List of work environment preferences
            - career_goals: List of career goals
            - skills_to_develop: List of skills to develop
            - preferred_project_types: List of preferred project types
            - company_mission_alignment: List of mission alignment preferences
            - preferred_company_size: List of preferred company sizes
            - funding_stage_preferences: List of preferred funding stages
            - total_compensation_expectations: Object with base_salary_min, base_salary_max, equity, and bonus
            - benefits_preferences: List of preferred benefits
            - deal_breakers: List of deal breakers
            - bad_experiences_to_avoid: List of bad experiences to avoid
            - willing_to_relocate: Boolean indicating if they're willing to relocate
            - preferred_interview_process: List of preferred interview process elements
            - company_reputation_importance: String describing importance of company reputation
            - preferred_management_style: List of preferred management styles
            - industries_to_explore: List of industries they're interested in exploring
            - project_visibility_preference: List of project visibility preferences"""

            # Create the user message with the transcript
            user_message = f"Please analyze this interview transcript and extract the required information. Return the information in a JSON object format:\n\n{transcript}"

            # Get completion from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            # Parse the response
            extracted_info = json.loads(response.choices[0].message.content)
            
            # Log the extracted information
            logger.info(f"📝 Extracted transcript information: {json.dumps(extracted_info, indent=2)}")
            
            return extracted_info

        except Exception as e:
            logger.error(f"Error extracting transcript information: {str(e)}")
            # Return empty structure if extraction fails
            return {
                "previous_companies": [],
                "tech_stack": [],
                "years_of_experience": 0,
                "industries": [],
                "undesired_industries": [],
                "company_size_at_join": 0,
                "current_company_size": 0,
                "company_stage": "",
                "experience_with_significant_company_growth": False,
                "early_stage_startup_experience": False,
                "leadership_experience": False,
                "preferred_work_arrangement": [],
                "preferred_locations": [],
                "visa_sponsorship_needed": False,
                "salary_expectations": {"min": 0, "max": 0},
                "desired_company_stage": [],
                "preferred_industries": [],
                "preferred_product_types": [],
                "motivation_for_job_change": [],
                "work_life_balance_preferences": "",
                "desired_company_culture": "",
                "traits_to_avoid_detected": [],
                "additional_notes": "",
                "candidate_tags": [],
                "next_steps": "",
                "role_preferences": [],
                "technologies_to_avoid": [],
                "company_culture_preferences": [],
                "work_environment_preferences": [],
                "career_goals": [],
                "skills_to_develop": [],
                "preferred_project_types": [],
                "company_mission_alignment": [],
                "preferred_company_size": [],
                "funding_stage_preferences": [],
                "total_compensation_expectations": {
                    "base_salary_min": 0,
                    "base_salary_max": 0,
                    "equity": "",
                    "bonus": ""
                },
                "benefits_preferences": [],
                "deal_breakers": [],
                "bad_experiences_to_avoid": [],
                "willing_to_relocate": False,
                "preferred_interview_process": [],
                "company_reputation_importance": "",
                "preferred_management_style": [],
                "industries_to_explore": [],
                "project_visibility_preference": []
            } 