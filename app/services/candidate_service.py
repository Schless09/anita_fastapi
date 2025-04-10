from typing import Dict, Any, Optional, List
import uuid
import base64
from datetime import datetime
from app.config.settings import Settings
from app.config.utils import get_table_name
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
from supabase._async.client import AsyncClient
from app.schemas.candidate import CandidateCreate, CandidateStatusUpdate

logger = logging.getLogger(__name__)

class CandidateService:
    def __init__(self, 
                 supabase_client: AsyncClient, 
                 retell_service: RetellService, 
                 openai_service: OpenAIService, 
                 settings: Settings):
        self.supabase: AsyncClient = supabase_client # Use injected client
        self.candidates_table_name = get_table_name("candidates", settings) # Use injected settings
        self.retell = retell_service # Use injected service
        self.openai = openai_service # Use injected service
        # Remove unused service assignments
        # self.vector_service = vector_service
        # self.matching = matching

    async def create_initial_candidate(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial candidate record with basic information."""
        logger.info(f"Creating initial candidate record for {submission_data['email']}")
        
        # Generate a new UUID for the candidate
        candidate_id = str(uuid.uuid4())
        
        now = datetime.utcnow()
        
        # Prepare initial candidate data, including all new fields
        insert_data = {
            "id": candidate_id,
            # Use correct field names based on schema/db
            "full_name": submission_data.get('name'),
            "email": submission_data.get('email'),
            "phone": submission_data.get('phone_number'), # Use DB column name 'phone'
            "linkedin_url": submission_data.get('linkedin'),
            # "resume_filename": submission_data.get('resume_filename'), # Removed: No DB column
            "work_environment": submission_data.get('work_environment'),
            "desired_locations": submission_data.get('desired_locations'),
            "preferred_sub_locations": submission_data.get('preferred_sub_locations'),
            "work_authorization": submission_data.get('work_authorization'),
            "visa_type": submission_data.get('visa_type'),
            "employment_types": submission_data.get('employment_types'),
            "availability": submission_data.get('availability'),
            "dream_role_description": submission_data.get('dream_role_description'),
            
            # Add the new consent fields
            "sms_consent": submission_data.get('sms_consent', False), # Default to False if not provided
            "legal_consent": submission_data.get('legal_consent', False), # Default to False if not provided
            
            # Existing fields
            "status": "submitted", # Initial status
            "status_last_updated": now.isoformat(),
            "profile_json": {}, # Initialize empty profile
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "is_resume_processed": False,
            "is_call_completed": False,
            "is_embedding_generated": False,
            "responsiveness_score": 0,
            "responsiveness_label": "Unresponsive"
            # Note: resume_url will be updated later after upload
        }

        # Remove keys with None values if the DB column doesn't handle them or has defaults
        insert_data = {k: v for k, v in insert_data.items() if v is not None}
        
        # --- DEBUG LOGGING --- 
        logger.info("Data prepared for Supabase insert.")
        # --- END DEBUG LOGGING ---

        try:
            # Insert the candidate record
            response = await self.supabase.table(self.candidates_table_name).insert(insert_data).execute()
            
            if not response.data:
                logger.error(f"Failed to create initial candidate record for {submission_data.get('email', 'N/A')}. DB Response: {response}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create candidate record"
                )
            
            logger.info(f"Successfully created initial candidate record: {candidate_id}")
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error creating initial candidate record for {submission_data.get('email', 'N/A')}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Log the data being inserted for debugging (be careful with sensitive info)
            # logger.debug(f"Insert data attempted: {insert_data}")
            raise HTTPException(
                status_code=500,
                detail=f"Error creating candidate record: {str(e)}"
            )

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

            response = await self.supabase.table(self.candidates_table_name)\
                .update(update_data)\
                .eq('id', candidate_id)\
                .execute()

            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error updating candidate profile {candidate_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating candidate profile: {str(e)}")

    async def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Basic method to fetch a candidate record.
        """
        try:
            response = await self.supabase.table(self.candidates_table_name)\
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

    async def get_candidate_profile_and_status(self, candidate_id: uuid.UUID) -> Dict[str, Any] | None:
        try:
            response = await self.supabase.table(self.candidates_table_name)\
                .select('profile_json, status')\
                .eq('id', str(candidate_id))\
                .single()\
                .execute()
            if response.data:
                 return response.data
            else:
                 logger.warning(f"(CandidateService) Candidate {candidate_id} not found.")
                 return None
        except Exception as e:
             logger.error(f"(CandidateService) Error fetching candidate profile and status for {candidate_id}: {e}")
             return None

    async def update_candidate_status_and_profile(self, update: CandidateStatusUpdate) -> Dict[str, Any] | None:
        update_fields = {
            'status': update.status,
            'updated_at': datetime.utcnow().isoformat()
        }

        try:
            response = await self.supabase.table(self.candidates_table_name)\
                .update(update_fields)\
                .eq('id', str(update.candidate_id))\
                .execute()
            if response.data:
                 logger.info(f"(CandidateService) Successfully updated candidate status and profile for {update.candidate_id}")
                 return response.data[0]
            else:
                 logger.warning(f"(CandidateService) Update candidate status and profile for {update.candidate_id} returned no data. Response: {response}")
                 return None
        except Exception as e:
             logger.error(f"(CandidateService) Error updating candidate status and profile for {update.candidate_id}: {e}")
             return None 

    async def update_resume_path(self, candidate_id: str, resume_path: str) -> Dict[str, Any]:
        """
        Update the resume_path field for a candidate.
        
        Args:
            candidate_id (str): The ID of the candidate to update
            resume_path (str): The path where the resume is stored
            
        Returns:
            Dict[str, Any]: The updated candidate record
        """
        logger.info(f"Updating resume URL for candidate {candidate_id} with path: {resume_path}")
        try:
            update_data = {
                # Use the correct column name from the DB schema
                "resume_url": resume_path, 
                "updated_at": datetime.utcnow().isoformat()
            }

            response = await self.supabase.table(self.candidates_table_name)\
                .update(update_data)\
                .eq('id', candidate_id)\
                .execute()

            if not response.data:
                logger.error(f"Failed to update resume URL for candidate {candidate_id}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to update candidate resume URL"
                )

            logger.info(f"Successfully updated resume URL for candidate {candidate_id}")
            return response.data[0]

        except Exception as e:
            logger.error(f"Error updating resume URL for candidate {candidate_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error updating candidate resume URL: {str(e)}"
            ) 