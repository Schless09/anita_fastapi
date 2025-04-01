import os
import asyncio
from app.config.settings import get_settings
from app.config.supabase import get_supabase_client

async def run_migrations():
    try:
        settings = get_settings()
        supabase = get_supabase_client()
        
        print("Running migrations...")
        
        # Enable pgvector extension (this needs to be done through the Supabase dashboard)
        print("\nNote: Please enable the pgvector extension through the Supabase dashboard if not already enabled.")
        
        # Create the jobs_dev table using the REST API
        print("\nCreating jobs_dev table...")
        
        # First, check if the table exists
        try:
            result = await supabase.table('jobs_dev').select("*").limit(1).execute()
            print("Table jobs_dev already exists.")
        except Exception as e:
            if 'relation "jobs_dev" does not exist' in str(e):
                # Create a minimal table first
                result = await supabase.table('jobs_dev').insert({
                    'job_title': 'test',
                    'job_url': 'test',
                    'positions_available': 1,
                    'hiring_urgency': 'test',
                    'seniority': 'test',
                    'work_arrangement': ['test'],
                    'location_city': ['test'],
                    'location_state': ['test'],
                    'location_country': 'test',
                    'visa_sponsorship': True,
                    'work_authorization': 'test',
                    'salary_range_min': 0,
                    'salary_range_max': 0,
                    'equity_range_min': 'test',
                    'equity_range_max': 'test',
                    'reporting_structure': 'test',
                    'team_structure': 'test',
                    'team_roles': ['test'],
                    'role_status': 'test',
                    'role_category': 'test',
                    'tech_stack_must_haves': ['test'],
                    'tech_stack_nice_to_haves': ['test'],
                    'tech_stack_tags': ['test'],
                    'tech_breadth': 'test',
                    'tech_breadth_requirement': 'test',
                    'minimum_years_of_experience': 0,
                    'domain_expertise': ['test'],
                    'ai_ml_exp_required': 'test',
                    'ai_ml_exp_focus': ['test'],
                    'infrastructure_experience': ['test'],
                    'system_design_expectation': 'test',
                    'coding_proficiency': 'test',
                    'languages': ['test'],
                    'version_control': ['test'],
                    'ci_cd_tools': ['test'],
                    'collaboration_tools': ['test'],
                    'leadership_required': True,
                    'education_required': 'test',
                    'education_advanced_degree': 'test',
                    'prior_startup_experience': True,
                    'startup_exp': 'test',
                    'advancement_history_required': True,
                    'career_trajectory': 'test',
                    'independent_work_capacity': 'test',
                    'independent_work': 'test',
                    'skills_must_have': ['test'],
                    'skills_preferred': ['test'],
                    'product_description': 'test',
                    'product_stage': 'test',
                    'product_dev_methodology': ['test'],
                    'product_technical_challenges': ['test'],
                    'product_development_stage': 'test',
                    'product_development_methodology': ['test'],
                    'key_responsibilities': ['test'],
                    'scope_of_impact': ['test'],
                    'expected_deliverables': ['test'],
                    'company_name': 'test',
                    'company_url': 'test',
                    'company_stage': 'test',
                    'company_funding_most_recent': 0,
                    'company_funding_total': 0,
                    'company_funding_investors': ['test'],
                    'company_founded': 'test',
                    'company_team_size': 0,
                    'company_mission': 'test',
                    'company_vision': 'test',
                    'company_growth_story': 'test',
                    'company_culture': 'test',
                    'company_scaling_plans': 'test',
                    'company_mission_and_impact': 'test',
                    'company_tech_innovation': 'test',
                    'company_industry_vertical': 'test',
                    'company_target_market': ['test'],
                    'ideal_companies': ['test'],
                    'deal_breakers': ['test'],
                    'disqualifying_traits': ['test'],
                    'culture_fit': ['test'],
                    'startup_mindset': ['test'],
                    'autonomy_level_required': 'test',
                    'growth_mindset': 'test',
                    'ideal_candidate_profile': 'test',
                    'interview_process_steps': ['test'],
                    'decision_makers': ['test'],
                    'recruiter_pitch_points': ['test']
                }).execute()
                print("Created jobs_dev table with initial schema.")
                
                # Delete the test record
                await supabase.table('jobs_dev').delete().eq('job_title', 'test').execute()
                print("Removed test record.")
            else:
                raise e
        
        print("\nMigrations completed!")
        print("\nIMPORTANT: Please enable the pgvector extension and add the embedding column through the Supabase dashboard:")
        print("1. Enable pgvector extension")
        print("2. Add column: ALTER TABLE jobs_dev ADD COLUMN embedding vector(1536);")
        
    except Exception as e:
        print(f"Error running migrations: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_migrations()) 