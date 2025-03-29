import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase import get_supabase_client

def test_supabase_connection():
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Test connection by fetching a single row from candidates_dev
        response = supabase.table('candidates_dev').select("*").limit(1).execute()
        
        print("✅ Successfully connected to Supabase!")
        print(f"Sample data: {response.data}")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {str(e)}")
        return False

if __name__ == "__main__":
    test_supabase_connection() 