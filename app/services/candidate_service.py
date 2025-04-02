from typing import Dict, Any, Optional, List
import uuid
import base64
from datetime import datetime
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.retell_service import RetellService
from app.services.openai_service import OpenAIService
# Remove unused services that are now handled by BrainAgent
# from app.services.vector_service import VectorService
# from app.services.matching_service import MatchingService
import logging
import tempfile
import os
import PyPDF2
import json
import asyncio
from fastapi import HTTPException
import traceback # Keep for potential error logging if needed elsewhere

settings = get_settings()
supabase = get_supabase_client()
retell = RetellService()
openai = OpenAIService()
# Remove unused service initializations
# vector_service = VectorService()
# matching = MatchingService()
logger = logging.getLogger(__name__)

class CandidateService:
    def __init__(self):
        self.supabase = supabase
        self.retell = retell
        self.openai = openai
        # Remove unused service assignments
        # self.vector_service = vector_service
        # self.matching = matching

    async def create_initial_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates the initial candidate record in the database.
        Extracts resume text if provided, but performs minimal processing.
        Does NOT generate embedding.
        Returns essential data like ID, email, and resume bytes for BrainAgent.
        """
        try:
            candidate_id = candidate_data.get('id', str(uuid.uuid4()))
            first_name = candidate_data.get('first_name', '')
            last_name = candidate_data.get('last_name', '')
            email = candidate_data.get('email', '')
            phone = candidate_data.get('phone', '')
            linkedin_url = candidate_data.get('linkedin_url')
            resume_content_bytes = candidate_data.get('resume_content') # Keep as bytes

            if not email:
                 raise ValueError("Email is required to create a candidate.")

            # Handle linkedin_url formatting
            if linkedin_url and linkedin_url.strip():
                if not linkedin_url.startswith(('http://', 'https://')):
                    linkedin_url = f"https://{linkedin_url}"
            else:
                linkedin_url = None

            # Basic profile structure
            profile_json = {
                'skills': candidate_data.get('skills', []),
                'current_role': 'pending', # Default placeholder
                'current_company': 'pending', # Default placeholder
                'education': [], 'experience': [], 'industries': [],
                'next_steps': '', 'tech_stack': [], 'career_goals': [],
                'company_stage': '', 'deal_breakers': [], 'candidate_tags': [],
                'additional_notes': '', 'role_preferences': [],
                'skills_to_develop': [], 'previous_companies': [], 'preferred_locations': [],
                'salary_expectations': {'max': 0, 'min': 0},
                'willing_to_relocate': False,
                'years_of_experience': 0,
                'processing_status': { # Minimal initial status
                    'status': 'submitted',
                    'last_updated': datetime.utcnow().isoformat(),
                    'resume_processed': False,
                    'call_completed': False,
                    'embedding_generated': False
                }
                # Add other fields initialized to defaults if needed...
            }

            # Extract resume text if content exists
            resume_text = ''
            if resume_content_bytes:
                try:
                    resume_text = await self._extract_text_from_pdf(resume_content_bytes)
                    profile_json['processing_status']['resume_found'] = True # Indicate resume was present
                except Exception as e:
                    logger.error(f"Error extracting resume text during initial create for {email}: {str(e)}")
                    profile_json['processing_status']['resume_found'] = False
                profile_json['resume_text'] = resume_text # Store extracted text (or empty string)

            # Prepare data for DB insert
            insert_data = {
                'id': candidate_id,
                'full_name': f"{first_name} {last_name}".strip(),
                'phone': phone,
                'email': email,
                'linkedin_url': linkedin_url,
                'profile_json': profile_json,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                # Do NOT store resume bytes in DB profile_json anymore
            }

            response = await self.supabase.table('candidates_dev').insert(insert_data).execute()
            if not response.data:
                 raise Exception(f"Database insert failed for candidate {email}")

            logger.info(f"Successfully created initial record for candidate {candidate_id} ({email})")

            # Return data needed for BrainAgent processing
            return {
                'id': candidate_id,
                'email': email,
                'resume_content_bytes': resume_content_bytes # Pass resume content for agent
            }

        except Exception as e:
            logger.error(f"Error creating initial candidate ({candidate_data.get('email', 'N/A')}): {str(e)}")
            # Consider specific exception types if needed
            raise HTTPException(status_code=500, detail=f"Error creating candidate: {str(e)}")

    # REMOVED: process_candidate_submission (replaced by create_initial_candidate + BrainAgent)
    # REMOVED: process_call_completion (functionality moved to BrainAgent.handle_call_processed)
    # REMOVED: _trigger_matchmaking (functionality moved to BrainAgent.handle_call_processed)
    # REMOVED: _generate_and_store_embedding (moved to BrainAgent)
    # REMOVED: _prepare_candidate_text_for_embedding (moved to BrainAgent)

    # --- Keep Essential Helper Methods --- 

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
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Clean up the temporary file
            os.unlink(temp_file_path)
            logger.info(f"Successfully extracted text from PDF (length: {len(text)})")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try: os.unlink(temp_file_path)
                except OSError: pass
            return ""  # Return empty string on failure

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

    # --- Potentially Keep Basic Update/Get Methods if Used by BrainAgent --- 
    # (These might be simplified or removed depending on BrainAgent's direct DB access)

    async def update_candidate_profile(self, candidate_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Basic method to update candidate fields (e.g., profile_json).
        """
        try:
            if 'updated_at' not in update_data:
                update_data['updated_at'] = datetime.utcnow().isoformat()

            response = await self.supabase.table('candidates_dev')\
                .update(update_data)\
                .eq('id', candidate_id)\
                .execute()

            if response.data:
                 logger.info(f"(CandidateService) Successfully updated profile for {candidate_id}")
                 return response.data[0]
            else:
                 logger.warning(f"(CandidateService) Update profile for {candidate_id} returned no data. Response: {response}")
                 return None
        except Exception as e:
             logger.error(f"(CandidateService) Error updating candidate profile for {candidate_id}: {e}")
             return None # Or raise?

    async def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Basic method to fetch a candidate record.
        """
        try:
            response = await self.supabase.table('candidates_dev')\
                .select('*')\
                .eq('id', candidate_id)\
                .maybe_single()\
                .execute()
            if response.data:
                 return response.data
            else:
                 logger.warning(f"(CandidateService) Candidate {candidate_id} not found.")
                 return None
        except Exception as e:
             logger.error(f"(CandidateService) Error fetching candidate {candidate_id}: {e}")
             return None # Or raise? 