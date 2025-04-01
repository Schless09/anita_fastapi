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
                    'first_name': candidate_data.get('first_name', ''),
                    'last_name': candidate_data.get('last_name', ''),
                    'email': candidate_data.get('email', ''),
                    'phone': candidate_data.get('phone', ''),
                    'linkedin_url': linkedin_url,
                    'current_role': current_role,
                    'current_company': current_company,
                    'processing_status': {
                        'resume_processed': False,
                        'call_completed': False,
                        'last_updated': datetime.utcnow().isoformat()
                    }
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
        Process call completion webhook from Retell.
        Updates candidate profile with call transcript information.
        Once both resume and call data are processed, generates and stores embedding.
        """
        try:
            call_id = call_data.get('call_id')
            candidate_id = call_data.get('metadata', {}).get('candidate_id')
            transcript = call_data.get('transcript')

            if not candidate_id:
                raise ValueError("No candidate_id found in webhook data")

            # Get current candidate data
            response = await self.supabase.table('candidates_dev').select('*').eq('id', candidate_id).single().execute()
            candidate = response.data
            if not candidate:
                raise ValueError(f"Candidate {candidate_id} not found")

            profile_json = candidate.get('profile_json', {})
            
            # Check if we've already processed this call
            call_data = profile_json.get('call_data', {})
            if call_data.get('call_id') == call_id and call_data.get('completed_at'):
                logger.info(f"Call {call_id} already processed for candidate {candidate_id}")
                return
            
            # Process transcript if available
            if transcript:
                # Extract information from transcript
                transcript_info = await self.openai.extract_transcript_information(transcript)
                
                # Update profile with transcript information
                profile_json.update(transcript_info)
                
                # Store raw transcript
                profile_json['call_data'] = {
                    'call_id': call_id,
                    'raw_transcript': transcript,
                    'completed_at': datetime.utcnow().isoformat()
                }
                
                # Update processing status
                profile_json['processing_status'] = {
                    'resume_processed': profile_json.get('processing_status', {}).get('resume_processed', False),
                    'call_completed': True,
                    'last_updated': datetime.utcnow().isoformat()
                }

                # Only generate and store embedding if both resume and call are processed
                if (profile_json.get('processing_status', {}).get('resume_processed') and 
                    profile_json.get('processing_status', {}).get('call_completed')):
                    try:
                        # Generate embedding from complete profile
                        text = self._prepare_text_for_embedding(profile_json)
                        vector = await self.openai.generate_embedding(text)
                        
                        # Store in vector database
                        await self.vector_service.upsert_candidate(
                            candidate_id,
                            profile_json
                        )
                        
                        # Update profile with embedding status
                        profile_json['processing_status']['embedding_complete'] = True
                        logger.info(f"Successfully generated and stored embedding for candidate {candidate_id}")
                    except Exception as e:
                        logger.error(f"Error generating embedding: {str(e)}")
                        profile_json['processing_status']['embedding_error'] = str(e)

                # Update candidate profile in Supabase
                await self.supabase.table('candidates_dev').update({
                    'profile_json': profile_json,
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', candidate_id).execute()

                logger.info(f"Successfully processed call completion for candidate {candidate_id}")

        except Exception as e:
            logger.error(f"Error processing call completion: {str(e)}")
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
            logger.info(f"Starting background processing for candidate {candidate_id}")
            
            # 1. Get candidate data
            response = await self.supabase.table('candidates_dev').select('*').eq('id', candidate_id).single().execute()
            candidate = response.data
            if not candidate:
                raise Exception(f"Candidate {candidate_id} not found")

            profile_json = candidate.get('profile_json', {})
            resume_data = profile_json.get('resume', {})
            
            # 2. Process resume if available
            if resume_data and resume_data.get('content'):
                try:
                    # Convert base64 back to bytes
                    resume_content = base64.b64decode(resume_data['content'])
                    
                    # Extract text and information from resume
                    extracted_info = await self.openai.extract_resume_information(resume_content)
                    
                    # Update profile with extracted information
                    profile_json.update(extracted_info)
                    
                    # Update processing status
                    profile_json['processing_status'] = {
                        'resume_processed': True,
                        'call_completed': False,
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    
                    # Update candidate profile
                    await self.supabase.table('candidates_dev').update({
                        'profile_json': profile_json,
                        'updated_at': datetime.utcnow().isoformat()
                    }).eq('id', candidate_id).execute()
                    
                    logger.info(f"Successfully processed resume for candidate {candidate_id}")

                except Exception as e:
                    logger.error(f"Error processing resume: {str(e)}")
                    # Update profile_json to indicate error
                    profile_json['processing_status'] = {
                        'resume_processed': False,
                        'resume_error': str(e),
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    await self.supabase.table('candidates_dev').update({
                        'profile_json': profile_json,
                        'updated_at': datetime.utcnow().isoformat()
                    }).eq('id', candidate_id).execute()
                    raise
            
            logger.info(f"Completed background processing for candidate {candidate_id}")
            
        except Exception as e:
            logger.error(f"Error in background processing for candidate {candidate_id}: {str(e)}")
            raise

    async def schedule_initial_contact(self, candidate_id: str) -> None:
        """
        Schedule initial contact with the candidate.
        """
        try:
            # Get candidate data
            response = await self.supabase.table('candidates_dev').select('*').eq('id', candidate_id).single().execute()
            candidate = response.data
            if not candidate:
                raise Exception(f"Candidate {candidate_id} not found")
            
            # Get phone number from candidate data
            phone = candidate.get('phone')
            if not phone:
                raise ValueError(f"Candidate {candidate_id} has no phone number")
            
            # Schedule call with Retell
            await self.retell.schedule_call(
                candidate_id=candidate_id,
                dynamic_variables={
                    'candidate_name': candidate.get('full_name', ''),
                    'current_role': candidate.get('profile_json', {}).get('current_role', ''),
                    'years_of_experience': candidate.get('profile_json', {}).get('years_of_experience', 0),
                    'phone': phone  # Add phone number to dynamic variables
                }
            )
            
            logger.info(f"Successfully scheduled initial contact for candidate {candidate_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling initial contact: {str(e)}")
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

    async def update_candidate_profile(self, candidate_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a candidate's profile in Supabase with new data.
        
        Args:
            candidate_id: The unique ID of the candidate to update
            update_data: Dictionary of fields to update
            
        Returns:
            The updated candidate data
        """
        try:
            logger.info(f"Updating candidate {candidate_id} in Supabase")
            
            # Check if candidate exists
            check_response = await self.supabase.table('candidates_dev').select('*').eq('id', candidate_id).execute()
            if not check_response.data:
                logger.error(f"Candidate {candidate_id} not found in Supabase")
                raise ValueError(f"Candidate {candidate_id} not found")
            
            # Add updated_at timestamp
            if 'updated_at' not in update_data:
                update_data['updated_at'] = datetime.utcnow().isoformat()
            
            # If updating profile_json, merge with existing rather than replacing
            existing_candidate = check_response.data[0]
            if 'profile_json' in update_data and existing_candidate.get('profile_json'):
                # Merge profile_json instead of replacing
                existing_profile = existing_candidate['profile_json']
                update_profile = update_data['profile_json']
                
                # Convert AIMessage objects to strings in update_profile
                if isinstance(update_profile, dict):
                    for key, value in update_profile.items():
                        if hasattr(value, 'content'):  # Check if it's an AIMessage
                            update_profile[key] = value.content
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if hasattr(subvalue, 'content'):
                                    update_profile[key][subkey] = subvalue.content
                
                # Deep merge the dictionaries
                merged_profile = self._deep_merge(existing_profile, update_profile)
                update_data['profile_json'] = merged_profile
            
            # Update candidate in database
            response = await self.supabase.table('candidates_dev').update(update_data).eq('id', candidate_id).execute()
            updated_candidate = response.data[0] if response.data else None
            
            if not updated_candidate:
                logger.error(f"Failed to update candidate {candidate_id}")
                raise Exception(f"Failed to update candidate {candidate_id}")
                
            logger.info(f"Successfully updated candidate {candidate_id}")
            return updated_candidate
            
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