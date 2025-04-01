import os
import asyncio
from app.config.settings import get_settings
from app.config.supabase import get_supabase_client

async def test_connection():
    try:
        settings = get_settings()
        supabase = get_supabase_client()
        
        # Test basic connection
        print("Testing Supabase connection...")
        result = await supabase.table('jobs_dev').select("*").limit(1).execute()
        print("Connection successful!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error connecting to Supabase: {str(e)}")
        print(f"Current settings:")
        print(f"SUPABASE_URL: {settings.supabase_url}")
        print(f"SUPABASE_KEY length: {len(settings.supabase_key) if settings.supabase_key else 0}")

if __name__ == "__main__":
    asyncio.run(test_connection()) 