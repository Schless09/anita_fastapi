import asyncio
from app.config.supabase import get_supabase_client
from app.config.settings import get_settings

async def test_supabase_connection():
    try:
        settings = get_settings()
        print(f"Supabase URL: {settings.supabase_url}")
        print(f"Supabase Key length: {len(settings.supabase_key)}")
        
        client = get_supabase_client()
        print("Supabase client created successfully")
        
        # Try to query the jobs table
        response = await client.from_("jobs_dev").select("*").limit(1).execute()
        print("Query successful:", response)
        
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    asyncio.run(test_supabase_connection()) 