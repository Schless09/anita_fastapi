import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from supabase import Client
import pinecone
from openai import OpenAI

logger = logging.getLogger(__name__)

class CustomerProfileService:
    def __init__(self, supabase: Client, pinecone_index: str, table_prefix: str = 'dev'):
        self.supabase = supabase
        self.pinecone_index = pinecone_index
        self.table_prefix = table_prefix
        self.openai = OpenAI()
    
    def _table(self, name: str) -> str:
        """Get the table name with the appropriate environment suffix"""
        return f"{name}-{self.table_prefix}"
    
    async def get_or_create_customer(self, email: str, phone: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
        """Get or create a customer profile."""
        try:
            # Check if customer exists
            result = self.supabase.table(self._table('customers')).select('*').eq('email', email).execute()
            
            if result.data:
                customer = result.data[0]
                # Update if new info provided
                if phone or name:
                    updates = {}
                    if phone:
                        updates['phone'] = phone
                    if name:
                        updates['name'] = name
                    if updates:
                        self.supabase.table(self._table('customers')).update(updates).eq('id', customer['id']).execute()
                        customer.update(updates)
            else:
                # Create new customer
                customer = self.supabase.table(self._table('customers')).insert({
                    'email': email,
                    'phone': phone,
                    'name': name
                }).execute().data[0]
            
            return customer
            
        except Exception as e:
            logger.error(f"Error in get_or_create_customer: {str(e)}")
            raise
    
    async def add_conversation(self, customer_id: str, channel: str, metadata: Dict = None) -> Dict[str, Any]:
        """Add a new conversation for a customer."""
        try:
            conversation = self.supabase.table(self._table('conversations')).insert({
                'customer_id': customer_id,
                'channel': channel,
                'metadata': metadata or {}
            }).execute().data[0]
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error in add_conversation: {str(e)}")
            raise
    
    async def add_message(self, conversation_id: str, content: str, sender: str, metadata: Dict = None) -> Dict[str, Any]:
        """Add a message to a conversation."""
        try:
            message = self.supabase.table(self._table('messages')).insert({
                'conversation_id': conversation_id,
                'content': content,
                'sender': sender,
                'metadata': metadata or {}
            }).execute().data[0]
            
            # Update customer vector after each message
            conversation = self.supabase.table(self._table('conversations')).select('customer_id').eq('id', conversation_id).execute().data[0]
            await self.update_customer_vector(conversation['customer_id'])
            
            return message
            
        except Exception as e:
            logger.error(f"Error in add_message: {str(e)}")
            raise
    
    async def add_call(self, customer_id: str, retell_call_id: str, status: str, **kwargs) -> Dict[str, Any]:
        """Add a call record."""
        try:
            call = self.supabase.table(self._table('calls')).insert({
                'customer_id': customer_id,
                'retell_call_id': retell_call_id,
                'status': status,
                'duration': kwargs.get('duration'),
                'recording_url': kwargs.get('recording_url'),
                'transcript': kwargs.get('transcript')
            }).execute().data[0]
            
            # Update vector if transcript available
            if kwargs.get('transcript'):
                await self.update_customer_vector(customer_id)
            
            return call
            
        except Exception as e:
            logger.error(f"Error in add_call: {str(e)}")
            raise
    
    async def update_customer_vector(self, customer_id: str):
        """Update the customer's vector in Pinecone based on all their interactions."""
        try:
            # Gather all customer data
            customer = self.supabase.table(self._table('customers')).select('*').eq('id', customer_id).execute().data[0]
            
            # Get resume
            resume = self.supabase.table(self._table('resumes')).select('*').eq('customer_id', customer_id).execute().data
            resume_text = resume[0]['raw_text'] if resume else ""
            
            # Get call transcripts
            calls = self.supabase.table(self._table('calls')).select('transcript').eq('customer_id', customer_id).execute().data
            call_texts = [call['transcript'] for call in calls if call['transcript']]
            
            # Get messages
            messages = self.supabase.table(self._table('messages')).select('content,sender').eq('customer_id', customer_id).execute().data
            message_texts = [f"{msg['sender']}: {msg['content']}" for msg in messages]
            
            # Combine all text
            combined_text = f"""
            Resume: {resume_text}
            
            Calls: {' '.join(call_texts)}
            
            Conversations: {' '.join(message_texts)}
            """
            
            # Create embedding
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text
            )
            vector = response.data[0].embedding
            
            # Update Pinecone
            pinecone.upsert(
                index_name=self.pinecone_index,
                vectors=[(customer_id, vector, {
                    'customer_id': customer_id,
                    'email': customer['email'],
                    'last_updated': datetime.utcnow().isoformat()
                })]
            )
            
            logger.info(f"Updated vector for customer {customer_id}")
            
        except Exception as e:
            logger.error(f"Error in update_customer_vector: {str(e)}")
            raise
    
    async def find_similar_candidates(self, job_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find candidates similar to a job posting."""
        try:
            # Get job vector
            job = pinecone.query(
                index_name='job-details',
                id=job_id
            )
            
            # Find matching candidates
            matches = pinecone.query(
                index_name=self.pinecone_index,
                vector=job.vector,
                top_k=top_k,
                include_metadata=True
            )
            
            # Enrich with detailed profiles
            candidates = []
            for match in matches.matches:
                customer = self.supabase.table(self._table('customers')).select('*').eq('id', match.id).execute().data[0]
                candidates.append({
                    'customer': customer,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in find_similar_candidates: {str(e)}")
            raise 