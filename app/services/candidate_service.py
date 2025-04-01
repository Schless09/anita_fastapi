from typing import Dict, Any, Optional, List
import uuid
import base64
from datetime import datetime
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.retell_service import RetellService
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.services.matching_service import MatchingService
import logging
import tempfile
import os
import PyPDF2
import json

settings = get_settings()
supabase = get_supabase_client()
retell = RetellService()
openai = OpenAIService()
vector_service = VectorService()
matching = MatchingService()
logger = logging.getLogger(__name__)

class CandidateService:
    def __init__(self):
        self.supabase = supabase
        self.retell = retell
        self.openai = openai
        self.vector_service = vector_service
        self.matching = matching

    async def process_candidate_submission(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new candidate submission from the frontend.
        Store basic candidate info in Supabase.
        Note: Retell call scheduling is now handled directly in the endpoint.
        """
        try:
            # 1. Get candidate ID from data or generate new one
            candidate_id = candidate_data.get('id', str(uuid.uuid4()))
            
            # Handle linkedin_url
            linkedin_url = candidate_data.get('linkedin_url')
            if linkedin_url and linkedin_url.strip():
                if not linkedin_url.startswith(('http://', 'https://')):
                    linkedin_url = f"https://{linkedin_url}"
            else:
                linkedin_url = None
            
            # Quick extract of current role/company from resume if available
            current_role = "current role"
            current_company = "current company"
            if resume_content := candidate_data.get('resume_content'):
                try:
                    # Quick extraction of current role/company
                    text = await self._extract_text_from_pdf(resume_content)
                    quick_info = await self.openai.quick_extract_current_position(text)
                    current_role = quick_info.get('current_role', 'current role')
                    current_company = quick_info.get('current_company', 'current company')
                except Exception as e:
                    logger.error(f"Error in quick resume extraction: {str(e)}")
            
            # Map frontend field names to database field names
            mapped_data = {
                'id': candidate_id,
                'full_name': f"{candidate_data.get('first_name', '')} {candidate_data.get('last_name', '')}".strip(),
                'phone': candidate_data.get('phone', ''),
                'email': candidate_data.get('email', ''),
                'linkedin_url': linkedin_url,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'profile_json': {
                    'current_role': current_role,
                    'current_company': current_company,
                    'skills': [],
                    'education': [],
                    'experience': [],
                    'professional_summary': '',
                    'additional_qualifications': [],
                    'resume_text': text # Store the processed resume text
                }
            }
            
            # Store resume in profile_json if available
            if resume_content := candidate_data.get('resume_content'):
                resume_filename = candidate_data.get('resume_filename', '')
                # Ensure filename is a PDF
                if not resume_filename.lower().endswith('.pdf'):
                    resume_filename = f"{resume_filename}.pdf"
                
                # Store resume content for later processing
                resume_base64 = base64.b64encode(resume_content).decode('utf-8')
                mapped_data['profile_json']['resume'] = {
                    'content': resume_base64,
                    'filename': resume_filename,
                    'content_type': 'application/pdf',
                    'size': len(resume_content),
                    'uploaded_at': datetime.utcnow().isoformat()
                }

            # Create candidate in database
            response = await self.supabase.table('candidates_dev').insert(mapped_data).execute()
            created_candidate = response.data[0]

            # Note: Retell call scheduling is now handled in the endpoint

            return {
                'id': candidate_id,
                'full_name': mapped_data['full_name'],
                'email': mapped_data['email'],
                'phone': mapped_data['phone'],
                'linkedin_url': linkedin_url,
                'created_at': mapped_data['created_at'],
                'updated_at': mapped_data['updated_at'],
                'profile_json': mapped_data['profile_json']
            }

        except Exception as e:
            raise Exception(f"Error processing candidate submission: {str(e)}")

    async def process_call_completion(self, call_data: Dict[str, Any]) -> None:
        """
        Process completed call data and update candidate profile.
        Extracts information from raw transcript using OpenAI.
        """
        try:
            # Extract essential data
            candidate_id = call_data.get('metadata', {}).get('candidate_id')
            transcript = call_data.get('transcript', '')

            if not candidate_id:
                logger.error("❌ No candidate_id found in call metadata")
                return

            if not transcript:
                logger.error("❌ No transcript found in call data")
                return
            
            # Get current profile
            response = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).single().execute()
            if not response.data:
                logger.error(f"❌ No profile found for candidate {candidate_id}")
                return
                
            current_profile = response.data.get('profile_json', {})
            
            # Extract information from transcript using OpenAI
            transcript_info = await self.openai.extract_transcript_info(transcript)
            
            # Combine all extracted information
            extracted_info = {
                # Resume information (preserve existing)
                "current_role": current_profile.get('current_role', ''),
                "current_company": current_profile.get('current_company', ''),
                "resume_text": current_profile.get('resume_text', ''),
                
                # Transcript information
                "previous_companies": transcript_info.get('previous_companies', []),
                "tech_stack": list(set(current_profile.get('tech_stack', []) + transcript_info.get('tech_stack', []))),
                "years_of_experience": transcript_info.get('years_of_experience', 0),
                "industries": transcript_info.get('industries', []),
                "undesired_industries": transcript_info.get('undesired_industries', []),
                "company_size_at_join": transcript_info.get('company_size_at_join', 0),
                "current_company_size": transcript_info.get('current_company_size', 0),
                "company_stage": transcript_info.get('company_stage', ''),
                "experience_with_significant_company_growth": transcript_info.get('experience_with_significant_company_growth', False),
                "early_stage_startup_experience": transcript_info.get('early_stage_startup_experience', False),
                "leadership_experience": transcript_info.get('leadership_experience', False),
                "preferred_work_arrangement": transcript_info.get('preferred_work_arrangement', []),
                "preferred_locations": transcript_info.get('preferred_locations', []),
                "visa_sponsorship_needed": transcript_info.get('visa_sponsorship_needed', False),
                "salary_expectations": transcript_info.get('salary_expectations', {'min': 0, 'max': 0}),
                "desired_company_stage": transcript_info.get('desired_company_stage', []),
                "preferred_industries": transcript_info.get('preferred_industries', []),
                "preferred_product_types": transcript_info.get('preferred_product_types', []),
                "motivation_for_job_change": transcript_info.get('motivation_for_job_change', []),
                "work_life_balance_preferences": transcript_info.get('work_life_balance_preferences', ''),
                "desired_company_culture": transcript_info.get('desired_company_culture', ''),
                "traits_to_avoid_detected": transcript_info.get('traits_to_avoid_detected', []),
                "additional_notes": transcript_info.get('additional_notes', ''),
                "candidate_tags": transcript_info.get('candidate_tags', []),
                "next_steps": transcript_info.get('next_steps', ''),
                "role_preferences": transcript_info.get('role_preferences', []),
                "technologies_to_avoid": transcript_info.get('technologies_to_avoid', []),
                "company_culture_preferences": transcript_info.get('company_culture_preferences', []),
                "work_environment_preferences": transcript_info.get('work_environment_preferences', []),
                "career_goals": transcript_info.get('career_goals', []),
                "skills_to_develop": transcript_info.get('skills_to_develop', []),
                "preferred_project_types": transcript_info.get('preferred_project_types', []),
                "company_mission_alignment": transcript_info.get('company_mission_alignment', []),
                "preferred_company_size": transcript_info.get('preferred_company_size', []),
                "funding_stage_preferences": transcript_info.get('funding_stage_preferences', []),
                "total_compensation_expectations": transcript_info.get('total_compensation_expectations', {
                    'base_salary_min': 0,
                    'base_salary_max': 0,
                    'equity': '',
                    'bonus': ''
                }),
                "benefits_preferences": transcript_info.get('benefits_preferences', []),
                "deal_breakers": transcript_info.get('deal_breakers', []),
                "bad_experiences_to_avoid": transcript_info.get('bad_experiences_to_avoid', []),
                "willing_to_relocate": transcript_info.get('willing_to_relocate', False),
                "preferred_interview_process": transcript_info.get('preferred_interview_process', []),
                "company_reputation_importance": transcript_info.get('company_reputation_importance', ''),
                "preferred_management_style": transcript_info.get('preferred_management_style', []),
                "industries_to_explore": transcript_info.get('industries_to_explore', []),
                "project_visibility_preference": transcript_info.get('project_visibility_preference', []),
                
                # Update processing status
                "processing_status": {
                    'resume_processed': current_profile.get('processing_status', {}).get('resume_processed', False),
                    'call_completed': True,
                    'last_updated': datetime.utcnow().isoformat()
                }
            }
            
            # Deep merge the new information with existing profile
            updated_profile = self._deep_merge(current_profile, extracted_info)
            
            # Update in Supabase
            update_data = {
                'profile_json': updated_profile,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            response = await self.supabase.table('candidates_dev').update(update_data).eq('id', candidate_id).execute()
            
            if not response.data:
                logger.error(f"❌ Failed to update profile for candidate {candidate_id}")
                raise Exception(f"Failed to update profile: No data returned from update")
            
            # Generate embeddings for matching
            await self.vector_service.generate_and_store_candidate_embedding(candidate_id)
            
            logger.info(f"✅ Successfully processed call completion for candidate {candidate_id}")

        except Exception as e:
            logger.error(f"❌ Error processing call completion: {str(e)}")
            raise

    def _prepare_text_for_embedding(self, profile_json: Dict[str, Any]) -> str:
        """
        Prepare profile data for embedding by converting it to a text string.
        """
        text_parts = []
        
        # Add basic information
        text_parts.append(f"Name: {profile_json.get('full_name', '')}")
        text_parts.append(f"Current Role: {profile_json.get('current_role', '')}")
        text_parts.append(f"Current Company: {profile_json.get('current_company', '')}")
        text_parts.append(f"Years of Experience: {profile_json.get('years_of_experience', '')}")
        
        # Add tech stack
        tech_stack = profile_json.get('tech_stack', [])
        if tech_stack:
            text_parts.append(f"Skills: {', '.join(tech_stack)}")
        
        # Add previous companies
        prev_companies = profile_json.get('previous_companies', [])
        if prev_companies:
            text_parts.append(f"Previous Companies: {', '.join(prev_companies)}")
        
        # Add education
        education = profile_json.get('education', [])
        if education:
            text_parts.append(f"Education: {', '.join(education)}")
        
        # Add career goals
        text_parts.append(f"Career Goals: {profile_json.get('career_goals', '')}")
        
        # Add work preferences
        work_prefs = profile_json.get('work_preferences', {})
        if work_prefs:
            text_parts.append(f"Work Preferences: Remote - {work_prefs.get('remote_preference', '')}, Location - {work_prefs.get('preferred_location', '')}")
        
        # Add industry preferences
        industries = profile_json.get('industry_preferences', [])
        if industries:
            text_parts.append(f"Industry Preferences: {', '.join(industries)}")
        
        return " | ".join(text_parts)

    async def extract_dynamic_variables(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract dynamic variables for Retell from candidate data.
        """
        # Implementation depends on what variables Retell needs
        return {
            'candidate_name': candidate_data.get('full_name', ''),
            'current_role': candidate_data.get('profile_json', {}).get('current_role', ''),
            'years_of_experience': candidate_data.get('profile_json', {}).get('years_of_experience', 0),
            # Add other variables as needed
        }

    async def process_candidate_background(self, candidate_id: str) -> None:
        """
        Process candidate data in the background.
        This includes:
        1. Extracting information from resume
        2. Storing in profile_json
        Note: Embedding generation is deferred until after call completion
        """
        try:
            # Get candidate data
            response = await self.supabase.table('candidates_dev').select('*').eq('id', candidate_id).single().execute()
            candidate = response.data
            if not candidate:
                raise ValueError(f"Candidate {candidate_id} not found")

            # Process resume
            resume_content = candidate.get('resume_content')
            if not resume_content:
                raise ValueError(f"No resume content found for candidate {candidate_id}")

            # Extract information from resume
            resume_info = await self.openai.extract_resume_info(resume_content)
            
            # Update profile with resume information
            profile_json = candidate.get('profile_json', {})
            profile_json.update({
                'resume_info': resume_info,
                'processing_status': {
                    'resume_processed': True,
                    'call_completed': False,
                    'last_updated': datetime.utcnow().isoformat()
                }
            })
            
            # Update candidate in Supabase
            await self.supabase.table('candidates_dev').update({
                'profile_json': profile_json,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', candidate_id).execute()
            
            logger.info(f"✅ Successfully processed resume for candidate {candidate_id}")
            
        except Exception as e:
            logger.error(f"Error processing candidate background: {str(e)}")
            raise

    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content.
        """
        try:
            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name

            # Read the PDF and extract text
            text = ""
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            # Clean up the temporary file
            os.unlink(temp_file_path)
            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""  # Return empty string if extraction fails

    async def update_candidate_profile(
        self,
        candidate_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a candidate's profile in Supabase.
        """
        try:
            # Update the candidate
            response = await self.supabase.table('candidates_dev').update(update_data).eq('id', candidate_id).execute()
            
            if not response.data:
                raise ValueError(f"Candidate {candidate_id} not found")
                
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error updating candidate profile: {str(e)}")
            raise
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Deep merge two dictionaries. If keys exist in both, dict2 values take precedence.
        For nested dictionaries, they are recursively merged.
        """
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result 

    async def manually_process_call(self, call_data: Dict[str, Any]) -> None:
        """
        Manually process a call for a candidate.
        This is used when the webhook processing fails.
        """
        try:
            logger.info(f"🔄 Manually processing call data")
            await self.process_call_completion(call_data)
            logger.info(f"✅ Successfully processed call data manually")
        except Exception as e:
            logger.error(f"Error manually processing call: {str(e)}")
            raise 