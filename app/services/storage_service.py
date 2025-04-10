import os
from typing import BinaryIO
import logging
from datetime import datetime
# import boto3  # Removed
# from botocore.client import Config # Removed
from supabase import create_client, Client, AClient # Added AClient
from app.config.settings import Settings

logger = logging.getLogger(__name__)

class StorageService:
    # Ensure supabase is type hinted as AsyncClient
    supabase: AClient 

    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize Supabase client with service role key for admin operations
        self.supabase = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
        
        # Removed S3 client initialization
        # self.s3_client = boto3.client(...)
        
        logger.info(f"Current environment: {settings.environment}")
        # Use dev bucket for both development and staging
        self.bucket_name = "resumes-prod" if settings.environment == "production" else "resumes-dev"
        logger.info(f"Selected bucket name: {self.bucket_name}")
        self._ensure_bucket_exists() # Keep this as it uses supabase client

    def _ensure_bucket_exists(self):
        """Ensure the resumes bucket exists, create if it doesn't."""
        try:
            buckets = self.supabase.storage.list_buckets()
            bucket_exists = any(bucket.name == self.bucket_name for bucket in buckets)
            
            if not bucket_exists:
                logger.info(f"Creating storage bucket: {self.bucket_name}")
                self.supabase.storage.create_bucket(
                    self.bucket_name,
                    options={
                        "public": False,
                        "allowed_mime_types": ["application/pdf"],
                        "file_size_limit": 50_000_000  # 50MB limit
                    }
                )
                logger.info(f"Successfully created bucket: {self.bucket_name}")
            else:
                # Update existing bucket configuration (Optional, but good practice)
                logger.info(f"Updating bucket configuration: {self.bucket_name}")
                self.supabase.storage.update_bucket(
                    self.bucket_name,
                    options={
                        "public": False,
                        "allowed_mime_types": ["application/pdf"],
                        "file_size_limit": 50_000_000  # 50MB limit
                    }
                )
                logger.info(f"Successfully updated bucket configuration: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {str(e)}")
            # Consider re-raising or handling more gracefully depending on requirements
            raise

    async def store_resume(self, file_content: BinaryIO, user_id: str, original_filename: str) -> str:
        """
        Store a resume file in Supabase storage.
        
        Args:
            file_content: The binary content of the file
            user_id: The ID of the user this resume belongs to
            original_filename: The original filename
            
        Returns:
            str: The path to the stored file (bucket_name/filename)
        """
        try:
            _, ext = os.path.splitext(original_filename)
            logger.info(f"Processing file with extension: {ext}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_{timestamp}{ext}"
            logger.info(f"Generated filename: {filename}")
            
            # Use Supabase storage API directly
            logger.info(f"Attempting Supabase storage upload for user {user_id} to {self.bucket_name}/{filename}")
            
            # Read file content into memory if it's a file object
            # Supabase client expects bytes or a file-like object 
            content_to_upload = file_content 
            if hasattr(file_content, 'read'):
                logger.info("Reading file content into memory for Supabase upload")
                content_bytes = file_content.read()
                # Check if content is already bytes
                if isinstance(content_bytes, bytes):
                     content_to_upload = content_bytes
                else:
                     logger.warning("File read did not return bytes, attempting fallback.")
                     # Reset if possible, although read() should consume it
                     if hasattr(file_content, 'seek'): 
                         file_content.seek(0)
                     # If not bytes, pass the original file object
                     content_to_upload = file_content 
            
            # Ensure file pointer is at the beginning if passing the file object
            if hasattr(content_to_upload, 'seek'):
                content_to_upload.seek(0)
                
            result = (
                self.supabase.storage
                .from_(self.bucket_name)
                .upload(
                    path=filename,
                    file=content_to_upload, # Use bytes or original file object
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            )
            
            file_path = f"{self.bucket_name}/{filename}"
            logger.info(f"Successfully stored resume using Supabase at {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error storing resume for user {user_id}: {str(e)}")
            # Consider more specific error handling for storage exceptions
            raise

    async def get_resume_url(self, file_path: str) -> str:
        """
        Get a temporary signed URL for accessing a stored resume using Supabase.
        
        Args:
            file_path: The path to the file (bucket_name/filename)
            
        Returns:
            str: Temporary signed URL for accessing the file
        """
        try:
            if '/' not in file_path:
                 raise ValueError(f"Invalid file_path format: {file_path}. Expected 'bucket/filename'.")
            
            bucket, filename = file_path.split('/', 1)
            logger.info(f"Generating signed URL for bucket '{bucket}', filename '{filename}'")
            
            # Use Supabase storage API directly
            result = await (
                self.supabase.storage
                .from_(bucket)
                .create_signed_url(
                    path=filename,
                    expires_in=3600, # URL expires in 1 hour
                    options={"download": False} # Set download=True if you want Content-Disposition header
                )
            )
            # Check if 'signedUrl' key exists and is not None
            signed_url = result.get('signedUrl')
            if not signed_url:
                 logger.error(f"Failed to generate signed URL. Supabase response: {result}")
                 raise Exception("Failed to generate signed URL from Supabase.")

            logger.info(f"Successfully generated signed URL for {file_path}")
            return signed_url
            
        except Exception as e:
            logger.error(f"Error getting resume URL for {file_path}: {str(e)}")
            raise 