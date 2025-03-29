from postgrest import AsyncPostgrestClient
from .settings import get_settings

def get_supabase_client() -> AsyncPostgrestClient:
    """Get a configured Supabase client instance with service role access."""
    settings = get_settings()
    
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError("Missing required Supabase configuration. Check SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    return AsyncPostgrestClient(
        base_url=f"{settings.supabase_url}/rest/v1",
        headers={
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}"
        }
    ) 