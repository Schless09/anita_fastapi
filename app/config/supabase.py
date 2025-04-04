from postgrest import AsyncPostgrestClient
from .settings import get_settings

def get_supabase_client() -> AsyncPostgrestClient:
    """Get a configured Supabase client instance.
    Uses service role key if available, otherwise falls back to anon key.
    """
    settings = get_settings()
    
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError("Missing required Supabase configuration. Check SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    # Prioritize service role key if available
    if settings.supabase_service_role_key:
        api_key = settings.supabase_service_role_key
        auth_header = f"Bearer {settings.supabase_service_role_key}"
    else:
        api_key = settings.supabase_key
        auth_header = f"Bearer {settings.supabase_key}"

    return AsyncPostgrestClient(
        base_url=f"{settings.supabase_url}/rest/v1",
        headers={
            "apikey": api_key,
            "Authorization": auth_header
        }
    ) 