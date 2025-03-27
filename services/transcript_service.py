import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional, List
from supabase import Client
from pinecone import Pinecone
from openai import OpenAI

logger = logging.getLogger(__name__)

class TranscriptService:
    def __init__(
        self,
        supabase: Client,
        pinecone_client: Pinecone,
        pinecone_index: str,
        openai_client: OpenAI
    ):
        self.supabase = supabase
        self.pc = pinecone_client
        self.index = pinecone_client.Index(pinecone_index)
        self.openai = openai_client
        self.table_name = 'calls-dev'  # Use dev table for development

    async def store_transcript(
        self,
        call_id: str,
        candidate_id: str,
        transcript: str,
        processed_data: Dict[str, Any],
        call_status: str,
        duration_ms: int,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store transcript data efficiently across different storage systems.
        Returns the stored data.
        """
        try:
            # 1. Store in Supabase (primary storage)
            supabase_data = {
                'candidate_id': candidate_id,
                'call_id': call_id,
                'transcript': transcript,
                'processed_data': processed_data,
                'call_status': call_status,
                'duration_ms': duration_ms,
                'processed_at': datetime.now().isoformat(),
                'email': metadata.get('email'),
                'name': metadata.get('name')
            }
            
            # Insert into call_transcripts
            result = self.supabase.table(self.table_name).insert(supabase_data).execute()
            logger.info("✅ Stored transcript in Supabase")

            # 2. Update candidate profile with latest call info
            if processed_data:
                candidate_update = {
                    'last_call_id': call_id,
                    'last_call_status': call_status,
                    'last_call_date': datetime.now().isoformat(),
                    'processed_data': processed_data
                }
                self.supabase.table('candidates-dev').update(candidate_update).eq('id', candidate_id).execute()
                logger.info("✅ Updated candidate profile")

            # 3. Store minimal data in Pinecone for vector search
            pinecone_metadata = {
                'candidate_id': candidate_id,
                'call_id': call_id,
                'processed_at': datetime.now().isoformat(),
                'call_status': call_status
            }
            
            # Create embedding for vector search
            embedding = await self._create_embedding(transcript)
            
            # Store in Pinecone
            self.index.upsert(vectors=[(
                call_id,
                embedding,
                pinecone_metadata
            )])
            logger.info("✅ Stored vector in Pinecone")

            return supabase_data

        except Exception as e:
            logger.error(f"❌ Error storing transcript: {str(e)}")
            raise

    async def update_email_status(self, call_id: str) -> None:
        """Update email sent status in both Supabase and Pinecone."""
        try:
            # Update Supabase
            self.supabase.table(self.table_name).update({
                'email_sent': True,
                'email_sent_at': datetime.utcnow().isoformat()
            }).eq('call_id', call_id).execute()
            
            # Update Pinecone metadata
            self.index.update(
                id=call_id,
                set_metadata={
                    'email_sent': True,
                    'email_sent_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info("✅ Updated email status")
        except Exception as e:
            logger.error(f"❌ Error updating email status: {str(e)}")
            raise

    async def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI API."""
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Error creating embedding: {str(e)}")
            raise 