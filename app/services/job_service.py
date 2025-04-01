from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.vector_service import VectorService
from app.services.openai_service import OpenAIService
from app.schemas.job_posting import JobPosting
import uuid
import json
import traceback

# Initialize services
settings = get_settings()
supabase = get_supabase_client()
vector_service = VectorService()
openai = OpenAIService()
logger = logging.getLogger(__name__)

class JobService:
    def __init__(self):
        self.supabase = supabase
        self.vector_service = vector_service
        self.openai = openai
        
    def _prepare_text_for_embedding(self, job_data: dict) -> str:
        """Prepare job data for embedding by combining relevant fields into a single text."""
        try:
            sections = [
                # Job Overview
                f"Job Title: {job_data['job_title']}",
                f"Seniority: {job_data['seniority']}",
                f"Work Arrangement: {', '.join(job_data['work_arrangement'])}",
                f"Location: {', '.join(job_data['location_city'])} ({', '.join(job_data['location_state'])}, {job_data['location_country']})",
                
                # Role Details
                f"Role Category: {job_data['role_category']}",
                f"Tech Breadth: {job_data['tech_breadth']}",
                f"Required Experience: {job_data['minimum_years_of_experience']} years",
                f"Domain Expertise: {', '.join(job_data['domain_expertise'])}",
                f"AI/ML Experience: {job_data['ai_ml_exp_required']} - {', '.join(job_data['ai_ml_exp_focus'])}",
                f"Infrastructure Experience: {', '.join(job_data['infrastructure_experience'])}",
                f"Coding Proficiency: {job_data['coding_proficiency']}",
                
                # Tech Stack
                f"Required Tech Stack: {', '.join(job_data['tech_stack_must_haves'])}",
                f"Preferred Tech Stack: {', '.join(job_data['tech_stack_nice_to_haves'])}",
                f"Languages: {', '.join(job_data['languages'])}",
                f"Version Control: {', '.join(job_data['version_control'])}",
                f"CI/CD Tools: {', '.join(job_data['ci_cd_tools'])}",
                f"Collaboration Tools: {', '.join(job_data['collaboration_tools'])}",
                
                # Education and Experience
                f"Education Required: {job_data['education_required']}",
                f"Advanced Degree: {job_data['education_advanced_degree']}",
                f"Prior Startup Experience: {'Required' if job_data['prior_startup_experience'] else 'Not Required'}",
                f"Career Trajectory: {job_data['career_trajectory']}",
                f"Independent Work: {job_data['independent_work']}",
                
                # Skills
                f"Required Skills: {', '.join(job_data['skills_must_have'])}",
                f"Preferred Skills: {', '.join(job_data['skills_preferred'])}",
                
                # Product
                f"Product Description: {job_data['product_description']}",
                f"Product Stage: {job_data['product_stage']}",
                f"Development Methodology: {', '.join(job_data['product_dev_methodology'])}",
                f"Technical Challenges: {', '.join(job_data['product_technical_challenges'])}",
                
                # Responsibilities
                f"Key Responsibilities: {', '.join(job_data['key_responsibilities'])}",
                f"Scope of Impact: {', '.join(job_data['scope_of_impact'])}",
                f"Expected Deliverables: {', '.join(job_data['expected_deliverables'])}",
                
                # Company
                f"Company: {job_data['company_name']}",
                f"Company Stage: {job_data['company_stage']}",
                f"Industry: {job_data['company_industry_vertical']}",
                f"Target Market: {', '.join(job_data['company_target_market'])}",
                f"Team Size: {job_data['company_team_size']}",
                f"Company Mission: {job_data['company_mission']}",
                f"Company Vision: {job_data['company_vision']}",
                f"Company Culture: {job_data['company_culture']}",
                
                # Candidate Fit
                f"Ideal Companies: {', '.join(job_data['ideal_companies'])}",
                f"Deal Breakers: {', '.join(job_data['deal_breakers'])}",
                f"Culture Fit: {', '.join(job_data['culture_fit'])}",
                f"Startup Mindset: {', '.join(job_data['startup_mindset'])}",
                f"Growth Mindset: {job_data['growth_mindset']}",
                f"Ideal Candidate Profile: {job_data['ideal_candidate_profile']}"
            ]
            
            return "\n".join(sections)
        except Exception as e:
            logger.error(f"Error preparing text for embedding: {str(e)}\n{traceback.format_exc()}")
            raise
        
    def _clean_text(self, value: Any) -> Optional[str]:
        """Clean and validate text input."""
        try:
            if value is None:
                return None
                
            # Convert to string if not already
            if not isinstance(value, str):
                logger.debug(f"Converting non-string value to string: {value} (type: {type(value)})")
                value = str(value)
            
            cleaned = value.strip()
            return cleaned if cleaned else None
            
        except Exception as e:
            logger.error(f"Error in _clean_text: {str(e)}, value: {value}, type: {type(value)}")
            return None
            
    def _ensure_array(self, value: Any) -> Optional[List[str]]:
        """Ensure value is converted to an array of strings."""
        try:
            if value is None:
                return None
                
            # If it's already a list
            if isinstance(value, list):
                logger.debug(f"Processing list value: {value}")
                # Filter out None and empty strings, and convert all items to strings
                cleaned = [str(item).strip() for item in value if item is not None and str(item).strip()]
                return cleaned if cleaned else None
                
            # If it's a string
            if isinstance(value, str):
                logger.debug(f"Processing string value: {value}")
                # Check if it's a comma-separated string
                if ',' in value:
                    items = [item.strip() for item in value.split(',') if item.strip()]
                    return items if items else None
                # Single string value
                cleaned = value.strip()
                return [cleaned] if cleaned else None
                
            # For other types (int, float, bool), convert to string and wrap in list
            logger.debug(f"Processing other type value: {value} (type: {type(value)})")
            return [str(value)]
            
        except Exception as e:
            logger.error(f"Error in _ensure_array: {str(e)}, value: {value}, type: {type(value)}")
            return None
        
    async def process_job_submission(self, job_data: dict) -> dict:
        """Process a new job submission."""
        try:
            # Log the incoming data structure
            logger.info(f"Processing job submission with data structure: {json.dumps(job_data, indent=2)}")
            
            # Validate the job data against our schema
            try:
                job_posting = JobPosting(**job_data)
                logger.info("Job data validation successful")
            except Exception as e:
                logger.error(f"Job data validation failed: {str(e)}\n{traceback.format_exc()}")
                raise
            
            # Generate an ID if not provided
            job_id = job_data.get('id', str(uuid.uuid4()))
            logger.info(f"Using job ID: {job_id}")
            
            # Prepare text for embedding
            try:
                text_for_embedding = self._prepare_text_for_embedding(job_data)
                logger.info("Successfully prepared text for embedding")
            except Exception as e:
                logger.error(f"Error preparing text for embedding: {str(e)}\n{traceback.format_exc()}")
                raise
            
            # Generate embedding
            try:
                embedding = await self.vector_service.generate_embedding(text_for_embedding)
                logger.info("Successfully generated embedding")
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}\n{traceback.format_exc()}")
                raise
            
            # Map job data to database schema
            db_job_data = {
                'id': job_id,
                'job_title': job_data['job_title'],
                'job_url': job_data['job_url'],
                'positions_available': job_data['positions_available'],
                'hiring_urgency': job_data['hiring_urgency'],
                'seniority': job_data['seniority'],
                'work_arrangement': job_data['work_arrangement'],
                'location_city': job_data['location_city'],
                'location_state': job_data['location_state'],
                'location_country': job_data['location_country'],
                'visa_sponsorship': job_data['visa_sponsorship'],
                'work_authorization': job_data['work_authorization'],
                'salary_range_min': job_data['salary_range_min'],
                'salary_range_max': job_data['salary_range_max'],
                'equity_range_min': job_data['equity_range_min'],
                'equity_range_max': job_data['equity_range_max'],
                'reporting_structure': job_data['reporting_structure'],
                'team_structure': job_data['team_structure'],
                'team_roles': job_data['team_roles'],
                'role_status': job_data['role_status'],
                'role_category': job_data['role_category'],
                'tech_stack_must_haves': job_data['tech_stack_must_haves'],
                'tech_stack_nice_to_haves': job_data['tech_stack_nice_to_haves'],
                'tech_stack_tags': job_data['tech_stack_tags'],
                'tech_breadth': job_data['tech_breadth'],
                'tech_breadth_requirement': job_data['tech_breadth_requirement'],
                'minimum_years_of_experience': job_data['minimum_years_of_experience'],
                'domain_expertise': job_data['domain_expertise'],
                'ai_ml_exp_required': job_data['ai_ml_exp_required'],
                'ai_ml_exp_focus': job_data['ai_ml_exp_focus'],
                'infrastructure_experience': job_data['infrastructure_experience'],
                'system_design_expectation': job_data['system_design_expectation'],
                'coding_proficiency': job_data['coding_proficiency'],
                'languages': job_data['languages'],
                'version_control': job_data['version_control'],
                'ci_cd_tools': job_data['ci_cd_tools'],
                'collaboration_tools': job_data['collaboration_tools'],
                'leadership_required': job_data['leadership_required'],
                'education_required': job_data['education_required'],
                'education_advanced_degree': job_data['education_advanced_degree'],
                'prior_startup_experience': job_data['prior_startup_experience'],
                'startup_exp': job_data['startup_exp'],
                'advancement_history_required': job_data['advancement_history_required'],
                'career_trajectory': job_data['career_trajectory'],
                'independent_work_capacity': job_data['independent_work_capacity'],
                'independent_work': job_data['independent_work'],
                'skills_must_have': job_data['skills_must_have'],
                'skills_preferred': job_data['skills_preferred'],
                'product_description': job_data['product_description'],
                'product_stage': job_data['product_stage'],
                'product_dev_methodology': job_data['product_dev_methodology'],
                'product_technical_challenges': job_data['product_technical_challenges'],
                'product_development_stage': job_data['product_development_stage'],
                'product_development_methodology': job_data['product_development_methodology'],
                'key_responsibilities': job_data['key_responsibilities'],
                'scope_of_impact': job_data['scope_of_impact'],
                'expected_deliverables': job_data['expected_deliverables'],
                'company_name': job_data['company_name'],
                'company_url': job_data['company_url'],
                'company_stage': job_data['company_stage'],
                'company_funding_most_recent': job_data['company_funding_most_recent'],
                'company_funding_total': job_data['company_funding_total'],
                'company_funding_investors': job_data['company_funding_investors'],
                'company_founded': job_data['company_founded'],
                'company_team_size': job_data['company_team_size'],
                'company_mission': job_data['company_mission'],
                'company_vision': job_data['company_vision'],
                'company_growth_story': job_data['company_growth_story'],
                'company_culture': job_data['company_culture'],
                'company_scaling_plans': job_data['company_scaling_plans'],
                'company_mission_and_impact': job_data['company_mission_and_impact'],
                'company_tech_innovation': job_data['company_tech_innovation'],
                'company_industry_vertical': job_data['company_industry_vertical'],
                'company_target_market': job_data['company_target_market'],
                'ideal_companies': job_data['ideal_companies'],
                'deal_breakers': job_data['deal_breakers'],
                'disqualifying_traits': job_data['disqualifying_traits'],
                'culture_fit': job_data['culture_fit'],
                'startup_mindset': job_data['startup_mindset'],
                'autonomy_level_required': job_data['autonomy_level_required'],
                'growth_mindset': job_data['growth_mindset'],
                'ideal_candidate_profile': job_data['ideal_candidate_profile'],
                'interview_process_steps': job_data['interview_process_steps'],
                'decision_makers': job_data['decision_makers'],
                'recruiter_pitch_points': job_data['recruiter_pitch_points'],
                'embedding': embedding,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Store in Supabase
            try:
                logger.info("Attempting to store job in Supabase")
                result = await self.supabase.table('jobs_dev').insert(db_job_data).execute()
                
                if not result.data:
                    logger.error(f"Error storing job in Supabase: No data returned")
                    raise Exception("Failed to store job: No data returned")
                
                logger.info(f"Successfully processed and stored job {job_id}")
                
                # Return a structured response
                return {
                    "status": "success",
                    "message": "Job submission processed and stored successfully",
                    "job": {
                        "id": job_id,
                        "title": job_data['job_title'],
                        "company": job_data['company_name'],
                        "location": {
                            "city": job_data['location_city'],
                            "state": job_data['location_state'],
                            "country": job_data['location_country']
                        },
                        "seniority": job_data['seniority'],
                        "work_arrangement": job_data['work_arrangement'],
                        "compensation": {
                            "salary_range": {
                                "min": job_data['salary_range_min'],
                                "max": job_data['salary_range_max']
                            },
                            "equity_range": {
                                "min": job_data['equity_range_min'],
                                "max": job_data['equity_range_max']
                            }
                        },
                        "created_at": db_job_data['created_at'],
                        "updated_at": db_job_data['updated_at']
                    }
                }
                
            except Exception as e:
                logger.error(f"Error storing job in Supabase: {str(e)}\n{traceback.format_exc()}")
                raise
            
        except Exception as e:
            logger.error(f"Error in process_job_submission: {str(e)}\n{traceback.format_exc()}")
            raise