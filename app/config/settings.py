from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from functools import lru_cache
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore' # Allow extra fields like SLACK_WEBHOOK_URL initially
    )

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.5
    
    # Retell
    retell_agent_id: str
    retell_api_key: str
    retell_api_base: str
    retell_webhook_url: str
    retell_from_number: str
    
    # Twilio
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    
    # Vercel
    vercel_protection_bypass: str
    
    # Slack
    slack_app_id: str
    slack_client_id: str
    slack_client_secret: str
    slack_signing_secret: str
    slack_verification_token: str
    slack_bot_token: str
    slack_webhook_url: Optional[str] = None
    
    # Ngrok
    ngrok_authtoken: str
    base_url: Optional[str] = None
    
    # Supabase
    supabase_url: str
    supabase_key: str
    supabase_service_role_key: str
    supabase_anon_key: str
    supabase_db_name: str = "postgres"

    # S3 Storage
    s3_endpoint: str = "https://izepykrdwrascjhxtxuz.supabase.co/storage/v1/s3"
    s3_access_key_id: str = "7c49ea376b8006a7205c440a39fbbaf5"
    s3_secret_access_key: str = "6dfd8431c0fdb5e26ca8b9b5508ca7e7ac45f53ed1806b55ef6770c374f551a4"
    s3_region: str = "us-east-1"

    # Email
    sender_email: str

    # Add development webhook URL
    development_webhook_url: Optional[str] = None  # Set in .env: DEVELOPMENT_WEBHOOK_URL=your-ngrok-url/webhook/retell

    # Candidate Processing
    min_call_duration_seconds: int = 300 # 5 minutes
    max_call_duration_seconds: int = 1800 # 30 minutes
    match_score_threshold: float = 0.50 # Threshold for sending match email
    max_matches_per_candidate: int = 10 # Max matches to store/consider

    @property
    def webhook_base_url(self) -> str:
        """Get the base URL for webhooks based on environment."""
        if self.environment == "development":
            # For local development, webhooks will be received by staging and forwarded
            return "https://anita-fastapi-staging.onrender.com"
        elif self.environment == "staging":
            return "https://anita-fastapi-staging.onrender.com"
        else:  # production
            return "https://anita-fastapi.onrender.com"
    
    @property
    def retell_webhook_url(self) -> str:
        """Get the webhook URL for Retell callbacks."""
        return "https://anita-fastapi.onrender.com/webhook/retell"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 