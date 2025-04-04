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
    
    # SendGrid
    sendgrid_api_key: Optional[str] = None
    sender_email: str
    sendgrid_inbound_hostname: Optional[str] = None
    sendgrid_webhook_url: Optional[str] = None
    
    # Vercel
    vercel_protection_bypass: Optional[str] = None
    
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

    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_candidates_index: Optional[str] = None
    pinecone_jobs_index: Optional[str] = None
    pinecone_call_status_index: Optional[str] = None

    @property
    def webhook_base_url(self) -> str:
        """Get the base URL for webhooks based on environment."""
        if self.environment == "production":
            return "https://anita-fastapi-2.vercel.app"
        else:
            return self.base_url  # This will be the ngrok URL in development
    
    @property
    def retell_webhook_url(self) -> str:
        """Get the Retell webhook URL based on environment."""
        return f"{self.webhook_base_url}/webhook/retell"
    
    @property
    def sendgrid_webhook_url(self) -> str:
        """Get the SendGrid webhook URL based on environment."""
        return f"{self.webhook_base_url}/email/webhook"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def get_table_name(base_name: str) -> str:
    """Returns the full table name with environment suffix (_dev or _prod)."""
    settings = get_settings()
    suffix = "_prod" if settings.environment == "production" else "_dev"
    return f"{base_name}{suffix}" 