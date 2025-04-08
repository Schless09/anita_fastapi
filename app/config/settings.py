from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    
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
    
    # Ngrok
    ngrok_authtoken: str
    base_url: str
    
    # Supabase
    supabase_url: str
    supabase_key: str
    supabase_service_role_key: str

    # S3 Storage
    s3_endpoint: str = "https://izepykrdwrascjhxtxuz.supabase.co/storage/v1/s3"
    s3_access_key_id: str = "7c49ea376b8006a7205c440a39fbbaf5"
    s3_secret_access_key: str = "6dfd8431c0fdb5e26ca8b9b5508ca7e7ac45f53ed1806b55ef6770c374f551a4"
    s3_region: str = "us-east-1"

    # Email
    sender_email: str

    @property
    def webhook_base_url(self) -> str:
        """Get the base URL for webhooks based on environment."""
        return "https://anita-fastapi.onrender.com"
    
    @property
    def retell_webhook_url(self) -> str:
        """Get the Retell webhook URL based on environment."""
        return f"{self.webhook_base_url}/webhook/retell/proxy"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 