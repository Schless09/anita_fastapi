import os
from typing import BinaryIO
import logging
from datetime import datetime
import boto3
from botocore.client import Config
from supabase import create_client, Client
from app.config.settings import Settings

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize Supabase client with service role key for admin operations
        self.supabase = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
        
        # Initialize S3 client for direct access
        self.s3_client = boto3.client(
            's3',
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
            config=Config(signature_version='s3v4')
        )
        
        logger.info(f"Current environment: {settings.environment}")
        # Use dev bucket for both development and staging
        self.bucket_name = "resumes-prod" if settings.environment == "production" else "resumes-dev"
        logger.info(f"Selected bucket name: {self.bucket_name}")
        self._ensure_bucket_exists()

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
                # Update existing bucket configuration
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
            raise

    async def store_resume(self, file_content: BinaryIO, user_id: str, original_filename: str) -> str:
        """
        Store a resume file in Supabase storage using S3.
        
        Args:
            file_content: The binary content of the file
            user_id: The ID of the user this resume belongs to
            original_filename: The original filename
            
        Returns:
            str: The path to the stored file
        """
        try:
            # Get file extension from original filename
            _, ext = os.path.splitext(original_filename)
            logger.info(f"Processing file with extension: {ext}")
            
            # Create a filename with user_id and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_{timestamp}{ext}"
            logger.info(f"Generated filename: {filename}")
            
            # Upload file to S3
            logger.info(f"Attempting S3 upload for user {user_id} to {self.bucket_name}/{filename}")
            
            try:
                # Read file content into memory if it's a file object
                if hasattr(file_content, 'read'):
                    logger.info("Reading file content into memory")
                    content = file_content.read()
                    # Reset file pointer for potential retry
                    if hasattr(file_content, 'seek'):
                        file_content.seek(0)
                else:
                    logger.info("Using provided content directly")
                    content = file_content

                logger.info("Initiating S3 put_object operation")
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType='application/pdf'
                )
                
                # Get the file path
                file_path = f"{self.bucket_name}/{filename}"
                logger.info(f"Successfully stored resume at {file_path}")
                
                return file_path
                
            except Exception as s3_error:
                logger.error(f"S3 upload error: {str(s3_error)}")
                logger.info("Attempting Supabase storage API fallback")
                
                try:
                    # Reset file pointer if possible
                    if hasattr(file_content, 'seek'):
                        file_content.seek(0)
                    
                    result = (
                        self.supabase.storage
                        .from_(self.bucket_name)
                        .upload(
                            path=filename,
                            file=file_content,
                            file_options={"content-type": "application/pdf", "upsert": "true"}
                        )
                    )
                    
                    file_path = f"{self.bucket_name}/{filename}"
                    logger.info(f"Successfully stored resume using Supabase fallback at {file_path}")
                    
                    return file_path
                except Exception as supabase_error:
                    logger.error(f"Supabase storage error: {str(supabase_error)}")
                    raise
            
        except Exception as e:
            logger.error(f"Error storing resume for user {user_id}: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e.__dict__)}")
            raise

    async def get_resume_url(self, file_path: str) -> str:
        """
        Get a temporary URL for accessing a stored resume.
        
        Args:
            file_path: The path to the file (bucket_name/filename)
            
        Returns:
            str: Temporary URL for accessing the file
        """
        try:
            bucket, filename = file_path.split('/', 1)
            
            try:
                # Generate pre-signed URL using S3 client
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket,
                        'Key': filename
                    },
                    ExpiresIn=3600  # URL expires in 1 hour
                )
                return url
                
            except Exception as s3_error:
                logger.error(f"S3 presigned URL error: {str(s3_error)}")
                # Fallback to Supabase storage API
                logger.info("Falling back to Supabase storage API for URL generation")
                result = (
                    self.supabase.storage
                    .from_(bucket)
                    .create_signed_url(
                        path=filename,
                        expires_in=3600,
                        options={"download": True}
                    )
                )
                return result['signedUrl']
            
        except Exception as e:
            logger.error(f"Error getting resume URL for {file_path}: {str(e)}")
            raise 