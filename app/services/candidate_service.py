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
from supabase._async.client import AsyncClient
from app.config.settings import get_table_name # Import from settings
from app.schemas.candidate import CandidateCreate, CandidateStatusUpdate

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
        self.supabase: AsyncClient = get_supabase_client()
        self.candidates_table_name = get_table_name("candidates") # Get dynamic table name
        self.retell = retell
        self.openai = openai
        # Remove unused service assignments
        # self.vector_service = vector_service
        # self.matching = matching

    async def create_candidate(self, submission: CandidateCreate) -> Dict[str, Any]:
        logger.info(f"Creating candidate with email: {submission.email}")
        # Build insert data dictionary
        insert_data = {
            "id": str(submission.id),  # Ensure id is string
            "name": submission.name,
            "email": submission.email,
            "phone": submission.phone_number, # Use corrected field name
            "linkedin": str(submission.linkedin) if submission.linkedin else None,
            "status": "submitted", # Initial status
            "profile_json": {}, # Initialize empty profile
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            # Add other default fields as necessary
            "is_resume_processed": False,
            "is_call_completed": False,
            "is_embedding_generated": False
        }
        # Note: Resume content is handled separately (upload to storage)

        try:
            response = await self.supabase.table(self.candidates_table_name).insert(insert_data).execute()
            if response.data:
                logger.info(f"Successfully created candidate: {submission.id}")
                # Add resume upload logic here if needed
                return response.data[0]
            else:
                logger.error(f"Failed to create candidate {submission.email}. Response: {response}")
                raise Exception(f"Database insert failed for candidate {submission.email}")
        except Exception as e:
            logger.error(f"Error creating candidate {submission.email}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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

            response = await self.supabase.table(self.candidates_table_name)\
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