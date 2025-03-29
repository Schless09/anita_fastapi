import sys
import os
from datetime import datetime
import uuid
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase import get_supabase_client

def insert_test_data():
    try:
        supabase = get_supabase_client()
        
        # Test candidate data matching the schema
        candidate_data = {
            'id': str(uuid.uuid4()),
            'full_name': 'John Developer',
            'email': 'john.dev@example.com',
            'phone': '+1234567890',
            'linkedin_url': 'https://linkedin.com/in/johndev',
            'github_url': 'https://github.com/johndev',
            'profile_json': {
                'current_role': 'Senior Software Engineer',
                'current_company': 'Tech Corp',
                'previous_companies': ['StartupX', 'BigTech Inc'],
                'tech_stack': ['Python', 'React', 'FastAPI', 'PostgreSQL', 'AWS'],
                'years_of_experience': 5,
                'industries': ['Software', 'FinTech'],
                'undesired_industries': ['Gaming', 'AdTech'],
                'company_size_at_join': 50,
                'current_company_size': 200,
                'company_stage': 'Series B',
                'experience_with_significant_company_growth': True,
                'early_stage_startup_experience': True,
                'leadership_experience': False,
                'preferred_work_arrangement': ['Remote', 'Hybrid'],
                'preferred_locations': ['San Francisco', 'New York'],
                'visa_sponsorship_needed': False,
                'salary_expectations': {
                    'min': 150000,
                    'max': 200000
                },
                'desired_company_stage': ['Series A', 'Series B', 'Series C'],
                'preferred_industries': ['AI/ML', 'Developer Tools', 'Enterprise Software'],
                'preferred_product_types': ['B2B SaaS', 'Developer Tools'],
                'motivation_for_job_change': ['Career Growth', 'Technical Challenges'],
                'work_life_balance_preferences': 'Flexible hours with focus on outcomes',
                'desired_company_culture': 'Engineering-driven with strong mentorship',
                'traits_to_avoid_detected': [],
                'additional_notes': 'Strong interest in AI/ML projects',
                'candidate_tags': ['senior-engineer', 'full-stack', 'remote-preferred'],
                'next_steps': 'Ready for technical interviews',
                'role_preferences': ['Tech Lead', 'Senior Engineer'],
                'technologies_to_avoid': ['PHP', 'WordPress'],
                'company_culture_preferences': ['Remote-first', 'Learning-focused'],
                'work_environment_preferences': ['Async communication', 'Documentation-driven'],
                'career_goals': ['Technical Leadership', 'Architecture Design'],
                'skills_to_develop': ['System Design', 'Team Leadership'],
                'preferred_project_types': ['Greenfield', 'Platform Development'],
                'company_mission_alignment': ['Developer Productivity', 'Technical Innovation'],
                'preferred_company_size': ['50-200', '201-1000'],
                'funding_stage_preferences': ['Series A', 'Series B', 'Series C'],
                'total_compensation_expectations': {
                    'base_salary_min': 150000,
                    'base_salary_max': 200000,
                    'equity': '0.1% - 1%',
                    'bonus': '10-15%'
                },
                'benefits_preferences': ['Health Insurance', 'Learning Budget', 'Home Office Setup'],
                'deal_breakers': ['No Remote Work', 'Legacy Tech Stack'],
                'bad_experiences_to_avoid': ['Micromanagement', 'No Code Reviews'],
                'willing_to_relocate': False,
                'preferred_interview_process': ['Technical Discussion', 'System Design', 'Code Review'],
                'company_reputation_importance': 'High',
                'preferred_management_style': ['Autonomous', 'Goal-Oriented'],
                'industries_to_explore': ['HealthTech', 'CleanTech'],
                'project_visibility_preference': ['High Impact', 'Customer Facing']
            },
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }

        # Test company data matching the schema
        company_data = {
            'id': str(uuid.uuid4()),
            'name': 'Example Tech Corp',
            'url': 'https://example.com',
            'stage': 'Series B',
            'team_size': '50-100',
            'founded': '2020',
            'mission': 'To revolutionize developer tooling',
            'vision': 'Make software development more accessible and efficient',
            'mission_and_impact': 'Creating tools that empower developers worldwide',
            'growth_story': 'Grew from 5 to 50 employees in 2 years',
            'culture': 'Remote-first, collaborative environment',
            'scaling_plans': 'Expanding to enterprise market',
            'tech_innovation': 'AI-powered code generation',
            'funding_most_recent_round': 'Series B',
            'funding_total': '$50M',
            'funding_investors': ['Accel', 'Sequoia', 'YC'],
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }

        # Test job data matching the schema
        job_data = {
            'id': str(uuid.uuid4()),
            'job_title': 'Senior Software Engineer',
            'job_url': 'https://example.com/jobs/123',
            'positions_available': '1-2',
            'hiring_urgency': 'High',
            'seniority_level': 'Senior',
            'work_arrangement': ['Remote', 'Hybrid'],
            'city': ['San Francisco', 'New York'],
            'state': ['CA', 'NY'],
            'country': 'USA',
            'visa_sponsorship': True,
            'work_authorization': 'US Citizens and Green Card Holders',
            'salary_min': '150000',
            'salary_max': '200000',
            'tech_stack_must_haves': ['Python', 'FastAPI', 'React'],
            'tech_stack_nice_to_haves': ['GraphQL', 'TypeScript'],
            'tech_stack_tags': ['Backend', 'Frontend', 'Full Stack'],
            'minimum_years_of_experience': '5+',
            'domain_expertise': ['Web Development', 'API Design'],
            'languages': ['Python', 'JavaScript', 'TypeScript'],
            'version_control': ['Git'],
            'ci_cd_tools': ['GitHub Actions', 'Jenkins'],
            'collaboration_tools': ['Slack', 'Jira'],
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        print("\nInserting test data into dev tables...")
        
        print("\nInserting candidate...")
        try:
            candidate_response = supabase.table('candidates_dev').insert(candidate_data).execute()
            print(f"✅ Candidate inserted: {candidate_response.data}")
            
            print("\nInserting company...")
            company_response = supabase.table('companies_dev').insert(company_data).execute()
            print(f"✅ Company inserted: {company_response.data}")
            
            # Add company_id to job data after company is created
            job_data['company_id'] = company_data['id']
            
            print("\nInserting job data...")
            job_response = supabase.table('jobs_dev').insert(job_data).execute()
            print(f"✅ Job inserted: {job_response.data}")

            # Test match data matching the schema
            match_data = {
                'id': str(uuid.uuid4()),
                'candidate_id': candidate_data['id'],
                'job_id': job_data['id'],
                'match_score': 0.85,
                'match_reason': 'Strong technical skill match and location preference alignment',
                'match_tags': [
                    'tech_stack_match',
                    'location_match',
                    'seniority_match',
                    'salary_match'
                ],
                'status': 'pending',
                'is_automatic_match': True,
                'next_step': 'Schedule initial interview',
                'matched_at': datetime.utcnow().isoformat(),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }

            print("\nInserting match data...")
            match_response = supabase.table('candidate_job_matches_dev').insert(match_data).execute()
            print(f"✅ Match inserted: {match_response.data}")

            # Test communication data matching the schema
            communication_data = {
                'id': str(uuid.uuid4()),
                'candidates_id': candidate_data['id'],
                'thread_id': str(uuid.uuid4()),
                'type': 'email',
                'direction': 'outbound',
                'subject': 'Interview Follow-up',
                'content': 'Thank you for your time during the interview...',
                'metadata': {
                    'email_id': 'em_123456',
                    'recipient': candidate_data['email'],
                    'status': 'delivered'
                },
                'timestamp': datetime.utcnow().isoformat()
            }

            print("\nInserting communication data...")
            communication_response = supabase.table('communications_dev').insert(communication_data).execute()
            print(f"✅ Communication inserted: {communication_response.data}")
            
        except Exception as e:
            print(f"❌ Error inserting data: {str(e)}")
            print("Candidate data attempted:", candidate_data)
            print("Company data attempted:", company_data)
            print("Job data attempted:", job_data)
            if 'match_data' in locals():
                print("Match data attempted:", match_data)
            if 'communication_data' in locals():
                print("Communication data attempted:", communication_data)
        
    except Exception as e:
        print(f"❌ Error in test data insertion: {str(e)}")

if __name__ == "__main__":
    insert_test_data() 