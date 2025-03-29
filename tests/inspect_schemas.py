import sys
import os
import asyncio
from pprint import pprint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.supabase import get_supabase_client

async def inspect_schemas():
    try:
        supabase = get_supabase_client()
        
        # Get list of all tables
        tables = [
            'candidates_dev',
            'candidates_prod',
            'jobs_dev',
            'jobs_prod',
            'candidate_job_matches_dev',
            'candidate_job_matches_prod',
            'communications_dev',
            'communications_prod'
        ]
        
        for table in tables:
            print(f"\n📋 Inspecting table: {table}")
            try:
                # Try to get table structure using a SELECT query
                response = await supabase.table(table).select("*").limit(1).execute()
                
                if response.data:
                    print("\nSample Data:")
                    pprint(response.data[0])
                else:
                    # If table is empty, try to get column names using a SELECT query
                    try:
                        # Try to get column names from the table definition
                        response = await supabase.table(table).select("").execute()
                        if hasattr(response, 'columns'):
                            print("\nColumns:")
                            for col in response.columns:
                                print(f"- {col}")
                        else:
                            print("Table exists but is empty and column information is not available")
                    except Exception as e:
                        print(f"Error getting column information: {str(e)}")
            
            except Exception as e:
                if "does not exist" in str(e):
                    print(f"Table does not exist: {str(e)}")
                else:
                    print(f"Error inspecting {table}: {str(e)}")
            
            print("-" * 80)
    
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {str(e)}")

if __name__ == "__main__":
    asyncio.run(inspect_schemas()) 