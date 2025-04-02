# Python script to insert a test job
import asyncio
import sys
import os
import uuid
from datetime import datetime

# Add project root to sys.path
project_root = os.getcwd() # Assumes running from workspace root
sys.path.insert(0, project_root)

try:
    from app.config.supabase import get_supabase_client
    from dotenv import load_dotenv
    load_dotenv() # Load .env
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

async def insert_test_job():
    print("Attempting to insert test job...")
    supabase = None
    try:
        supabase = get_supabase_client()
        test_job_data = {
            # Minimal required fields based on schema (assuming NOT NULL constraints where applicable)
            # Adjust these based on actual non-nullable fields if the script fails
            # 'id': uuid.uuid4().hex, # REMOVED: Cannot insert ID for GENERATED ALWAYS column
            'job_title': "Webhook Test Job - Delete Me",
            'job_url': f"http://example.com/test/{uuid.uuid4()}",
            'positions_available': 1,
            'hiring_urgency': "TEST",
            'seniority': "TEST",
            'work_arrangement': ["TEST"],
            'location_city': ["TEST"],
            'location_state': ["TEST"],
            'location_country': "TEST",
            'visa_sponsorship': False,
            'work_authorization': "TEST",
            'salary_range_min': 0,
            'salary_range_max': 0,
            'equity_range_min': "0",
            'equity_range_max': "0",
            'reporting_structure': "TEST",
            'team_structure': "TEST",
            'team_roles': ["TEST"],
            'role_status': "TEST",
            'role_category': ["TEST"],
            'tech_stack_must_haves': ["TEST"],
            'tech_stack_nice_to_haves': ["TEST"],
            'tech_stack_tags': ["TEST"],
            'minimum_years_of_experience': 0,
            'domain_expertise': ["TEST"],
            'ai_ml_exp_required': "TEST",
            'ai_ml_exp_focus': ["TEST"],
            'infrastructure_experience': ["TEST"],
            'system_design_expectation': "TEST",
            'coding_proficiency': "TEST",
            'languages': ["TEST"],
            'version_control': ["TEST"],
            'ci_cd_tools': ["TEST"],
            'collaboration_tools': ["TEST"],
            'leadership_required': False,
            'education_required': "TEST",
            'education_advanced_degree': "TEST",
            'prior_startup_experience': False,
            'startup_exp': "TEST",
            'career_trajectory': "TEST",
            'independent_work_capacity': "TEST",
            'independent_work': "TEST",
            'skills_must_have': ["TEST"],
            'skills_preferred': ["TEST"],
            'product_description': "TEST",
            'product_stage': "TEST",
            'product_dev_methodology': ["TEST"],
            'product_technical_challenges': ["TEST"],
            'key_responsibilities': ["TEST"],
            'scope_of_impact': ["TEST"],
            'expected_deliverables': ["TEST"],
            'company_name': "TEST Inc.",
            'company_url': "http://example.com/test",
            'company_stage': "TEST",
            'company_funding_most_recent': 0,
            'company_funding_total': 0,
            'company_funding_investors': ["TEST"],
            'company_founded': "TEST",
            'company_team_size': 1,
            'company_mission': "TEST",
            'company_vision': "TEST",
            'company_growth_story': "TEST",
            'company_culture': "TEST",
            'company_scaling_plans': "TEST",
            'company_tech_innovation': "TEST",
            'company_industry_vertical': ["TEST"],
            'company_target_market': ["TEST"],
            'ideal_companies': ["TEST"],
            'deal_breakers': ["TEST"],
            'disqualifying_traits': ["TEST"],
            'culture_fit': ["TEST"],
            'startup_mindset': ["TEST"],
            'autonomy_level_required': "TEST",
            'growth_mindset': "TEST",
            'ideal_candidate_profile': "TEST",
            'interview_process_steps': ["TEST"],
            'decision_makers': ["TEST"],
            'recruiter_pitch_points': ["TEST"],
            # embedding and embedding_metadata should be NULL initially
        }
        # print(f"Inserting job with ID: {test_job_data['id']}") # Cannot print ID before insert
        print("Inserting test job data...")
        response = await supabase.table('jobs_dev').insert(test_job_data).execute()
        print(f"Insertion response: {response}")
        if response.data:
            print("Test job inserted successfully!")
            print(f"Job ID: {response.data[0].get('id')}") # Log the actual ID returned by DB if needed
        else:
            print(f"Insertion failed or returned no data. Response: {response}")

    except Exception as e:
        print(f"Error inserting test job: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Client closing logic if needed (depends on supabase-py version)
        pass

if __name__ == "__main__":
    asyncio.run(insert_test_job()) 