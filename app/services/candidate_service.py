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
import asyncio
from fastapi import HTTPException

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
                    'skills': [],
                    'education': [],
                    'experience': [],
                    'industries': [],
                    'next_steps': '',
                    'tech_stack': [],
                    'resume_text': text if 'text' in locals() else '',
                    'career_goals': [],
                    'current_role': current_role,
                    'company_stage': '',
                    'deal_breakers': [],
                    'candidate_tags': [],
                    'current_company': current_company,
                    'additional_notes': '',
                    'role_preferences': [],
                    'processing_status': {
                        'last_updated': datetime.utcnow().isoformat(),
                        'call_completed': False,
                        'resume_processed': False
                    },
                    'skills_to_develop': [],
                    'previous_companies': [],
                    'preferred_locations': [],
                    'salary_expectations': {
                        'max': 0,
                        'min': 0
                    },
                    'willing_to_relocate': False,
                    'years_of_experience': 0,
                    'benefits_preferences': [],
                    'company_size_at_join': 0,
                    'current_company_size': 0,
                    'preferred_industries': [],
                    'professional_summary': '',
                    'undesired_industries': [],
                    'desired_company_stage': [],
                    'industries_to_explore': [],
                    'leadership_experience': False,
                    'technologies_to_avoid': [],
                    'preferred_company_size': [],
                    'desired_company_culture': '',
                    'preferred_product_types': [],
                    'preferred_project_types': [],
                    'visa_sponsorship_needed': False,
                    'bad_experiences_to_avoid': [],
                    'traits_to_avoid_detected': [],
                    'additional_qualifications': [],
                    'company_mission_alignment': [],
                    'funding_stage_preferences': [],
                    'motivation_for_job_change': [],
                    'preferred_management_style': [],
                    'preferred_work_arrangement': [],
                    'company_culture_preferences': [],
                    'preferred_interview_process': [],
                    'work_environment_preferences': [],
                    'company_reputation_importance': '',
                    'project_visibility_preference': [],
                    'work_life_balance_preferences': '',
                    'early_stage_startup_experience': False,
                    'total_compensation_expectations': {
                        'bonus': '',
                        'equity': '',
                        'base_salary_max': 0,
                        'base_salary_min': 0
                    },
                    'experience_with_significant_company_growth': False
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
        """
        try:
            # Extract candidate_id from metadata
            candidate_id = call_data.get('metadata', {}).get('candidate_id')
            if not candidate_id:
                logger.error("No candidate_id found in call metadata")
                return

            # Get transcript
            transcript = call_data.get('transcript')
            if not transcript:
                logger.error("No transcript found in call data")
                return

            # Get current candidate profile
            response = await self.supabase.table('candidates_dev').select('profile_json').eq('id', candidate_id).execute()
            if not response.data:
                logger.error(f"Could not find candidate profile for {candidate_id}")
                return

            current_profile = response.data[0].get('profile_json', {})
            
            # Extract information from transcript using OpenAI
            extracted_info = await self.openai.extract_transcript_info(transcript)
            
            # Only merge if we got valid extracted info
            if not extracted_info:
                logger.error(f"No valid information extracted from transcript for candidate {candidate_id}")
                return
                
            # Deep merge the extracted information with current profile, but be careful about empty values
            merged_profile = self._deep_merge(current_profile, extracted_info)
            
            # Update processing status
            merged_profile['processing_status'] = {
                'status': 'completed',
                'last_updated': datetime.utcnow().isoformat(),
                'call_completed': True
            }
            
            # Update candidate in Supabase with retry logic
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    # Update only the profile_json field
                    update_data = {
                        'profile_json': merged_profile,
                        'updated_at': datetime.utcnow().isoformat()
                    }
                    
                    response = await self.supabase.table('candidates_dev').update(update_data).eq('id', candidate_id).execute()
                    
                    if response.data:
                        logger.info(f"✅ Successfully processed call completion for candidate {candidate_id}")
                        # Trigger matchmaking after successful profile update
                        await self._trigger_matchmaking(candidate_id, merged_profile)
                        break
                    else:
                        logger.error(f"Failed to update candidate profile: {candidate_id}")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed to update candidate profile: {str(e)}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to update candidate profile after {max_retries} attempts: {str(e)}")
                        raise

        except Exception as e:
            logger.error(f"Error processing call completion: {str(e)}")
            raise

    async def _trigger_matchmaking(self, candidate_id: str, profile: Dict[str, Any]) -> None:
        """
        Trigger matchmaking process for a candidate after their profile is updated.
        """
        try:
            logger.info(f"Starting matchmaking process for candidate {candidate_id}")
            
            # Prepare candidate data for matching
            candidate_data = {
                "id": candidate_id,
                "profile_json": profile
            }
            
            # Get job matches
            matches = await self.matching.match_candidate_to_jobs(candidate_data)
            
            if matches:
                logger.info(f"Found {len(matches)} potential job matches for candidate {candidate_id}")
                
                # Store matches in the database
                for match in matches:
                    match_data = {
                        "candidate_id": candidate_id,
                        "job_id": match["job_id"],
                        "similarity_score": match["similarity"],
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    await self.supabase.table('candidate_job_matches').insert(match_data).execute()
                
                logger.info(f"✅ Successfully stored {len(matches)} job matches for candidate {candidate_id}")
            else:
                logger.info(f"No job matches found for candidate {candidate_id}")
                
        except Exception as e:
            logger.error(f"Error in matchmaking process for candidate {candidate_id}: {str(e)}")
            # Don't raise the exception - we don't want to fail the profile update if matchmaking fails

    def _prepare_text_for_embedding(self, profile: Dict[str, Any]) -> str:
        """
        Prepare text for embedding generation by combining relevant profile information.
        Also updates embedding_metadata to track what information was included.
        """
        # Initialize embedding metadata if it doesn't exist
        if 'embedding_metadata' not in profile:
            profile['embedding_metadata'] = {
                'last_updated': datetime.utcnow().isoformat(),
                'sources': [],
                'fields_included': []
            }
        
        # Track which fields we're including
        fields_included = []
        text_parts = []
        
        # Add current role and company
        if profile.get('current_role') and profile['current_role'] != "current role":
            text_parts.append(f"Current role: {profile['current_role']}")
            fields_included.append('current_role')
        if profile.get('current_company') and profile['current_company'] != "current company":
            text_parts.append(f"Current company: {profile['current_company']}")
            fields_included.append('current_company')
        
        # Add years of experience
        if profile.get('years_of_experience'):
            text_parts.append(f"Years of experience: {profile['years_of_experience']}")
            fields_included.append('years_of_experience')
        
        # Add skills and tech stack
        if profile.get('skills'):
            text_parts.append(f"Skills: {', '.join(profile['skills'])}")
            fields_included.append('skills')
        if profile.get('tech_stack'):
            text_parts.append(f"Tech stack: {', '.join(profile['tech_stack'])}")
            fields_included.append('tech_stack')
        
        # Add experience
        if profile.get('experience'):
            exp_text = "Experience: "
            for exp in profile['experience']:
                if exp.get('title') and exp.get('company'):
                    exp_text += f"{exp['title']} at {exp['company']}, "
            text_parts.append(exp_text.rstrip(', '))
            fields_included.append('experience')
        
        # Add education
        if profile.get('education'):
            edu_text = "Education: "
            for edu in profile['education']:
                if edu.get('degree') and edu.get('institution'):
                    edu_text += f"{edu['degree']} from {edu['institution']}, "
            text_parts.append(edu_text.rstrip(', '))
            fields_included.append('education')
        
        # Add career goals
        if profile.get('career_goals'):
            text_parts.append(f"Career goals: {', '.join(profile['career_goals'])}")
            fields_included.append('career_goals')
        
        # Add work preferences
        if profile.get('work_preferences'):
            prefs = profile['work_preferences']
            if prefs.get('arrangement'):
                text_parts.append(f"Work arrangement preferences: {', '.join(prefs['arrangement'])}")
                fields_included.append('work_preferences.arrangement')
            if prefs.get('benefits'):
                text_parts.append(f"Benefits preferences: {', '.join(prefs['benefits'])}")
                fields_included.append('work_preferences.benefits')
            if prefs.get('company_size'):
                text_parts.append(f"Company size preferences: {', '.join(prefs['company_size'])}")
                fields_included.append('work_preferences.company_size')
        
        # Add industry preferences
        if profile.get('preferred_industries'):
            text_parts.append(f"Preferred industries: {', '.join(profile['preferred_industries'])}")
            fields_included.append('preferred_industries')
        
        # Add project preferences
        if profile.get('preferred_project_types'):
            text_parts.append(f"Preferred project types: {', '.join(profile['preferred_project_types'])}")
            fields_included.append('preferred_project_types')
        
        # Update embedding metadata
        profile['embedding_metadata'].update({
            'last_updated': datetime.utcnow().isoformat(),
            'fields_included': fields_included
        })
        
        # Add source information if available
        if profile.get('call_status'):
            profile['embedding_metadata']['sources'].append({
                'type': 'call',
                'timestamp': datetime.utcnow().isoformat(),
                'completeness': profile['call_status'].get('is_complete', False)
            })
        
        # Combine all text parts
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
        Empty values in dict2 will not overwrite existing values in dict1.
        """
        result = dict1.copy()
        for key, value in dict2.items():
            # Skip empty values
            if value is None or (isinstance(value, (list, dict)) and not value):
                continue
                
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                # Only update if the new value is not empty
                if isinstance(value, str) and value.strip():
                    result[key] = value
                elif isinstance(value, (list, dict)) and value:
                    result[key] = value
                elif isinstance(value, (int, float)) and value != 0:
                    result[key] = value
                elif isinstance(value, bool):
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

    async def process_call_analyzed(self, candidate_id: str, call_id: str, transcript: str) -> Dict[str, Any]:
        """
        Process a call that has been analyzed and update the candidate's profile.
        Handles partial information and incomplete calls gracefully.
        """
        try:
            # Get the candidate
            candidate = await self.get_candidate(candidate_id)
            if not candidate:
                raise HTTPException(status_code=404, detail="Candidate not found")

            # Extract information from transcript
            transcript_info = await self.openai.extract_transcript_info(transcript)
            
            # Update the candidate's profile with any available information
            if transcript_info:
                # Initialize profile_json if it doesn't exist
                if not candidate.profile_json:
                    candidate.profile_json = {}
                
                # Update only the fields that have values
                for key, value in transcript_info.items():
                    if value and key != 'call_status':  # Skip empty values and call_status
                        if isinstance(value, list) and value:  # For list fields, only update if not empty
                            candidate.profile_json[key] = value
                        elif isinstance(value, dict) and any(value.values()):  # For dict fields, only update if any values
                            candidate.profile_json[key] = value
                        elif isinstance(value, str) and value.strip():  # For string fields, only update if not empty
                            candidate.profile_json[key] = value

                # Update call status
                if 'call_status' in transcript_info:
                    candidate.profile_json['call_status'] = transcript_info['call_status']
                
                # Update the candidate
                await self.update_candidate(candidate_id, candidate)
                
                logger.info(f"Successfully updated candidate {candidate_id} with transcript information")
                return {"status": "success", "message": "Candidate profile updated with transcript information"}
            else:
                logger.warning(f"No information could be extracted from transcript for candidate {candidate_id}")
                return {"status": "warning", "message": "No information could be extracted from transcript"}
                
        except Exception as e:
            logger.error(f"Error processing call analyzed for candidate {candidate_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 