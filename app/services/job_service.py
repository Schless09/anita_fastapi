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
import re # Added for _safe_int helper

# Initialize services
settings = get_settings()
supabase = get_supabase_client()
vector_service = VectorService()
openai = OpenAIService()
logger = logging.getLogger(__name__)

def _safe_int(value, default=None):
    """Safely convert a value to an integer, handling common non-numeric strings."""
    if value is None:
        return default
    try:
        # Direct conversion first
        return int(value)
    except (ValueError, TypeError):
        # Handle strings like '5+', '1-2', etc. by extracting leading digits
        if isinstance(value, str):
            match = re.match(r'^(\d+)', value)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, TypeError):
                    pass # Fall through to default if extraction fails
        logger.warning(f"Could not convert value '{value}' to int, using default '{default}'")
        return default

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
                f"Role Category: {', '.join(job_data['role_category'])}",
                f"Seniority: {job_data['seniority']}",
                f"Work Arrangement: {', '.join(job_data['work_arrangement'])}",
                f"Role Status: {job_data['role_status']}",
                f"Tech Stack Must Haves: {', '.join(job_data['tech_stack_must_haves'])}",
                f"Tech Stack Nice to Haves: {', '.join(job_data['tech_stack_nice_to_haves'])}",
                f"Tech Stack Tags: {', '.join(job_data['tech_stack_tags'])}",
                f"Minimum Years of Experience: {job_data['minimum_years_of_experience']}",
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
        """Ensure the value is a list of strings."""
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        if isinstance(value, str):
            # Simple split by comma for basic string-to-list conversion, adjust if needed
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(value)] # Wrap single non-list items in a list

    async def process_job_submission(self, job_data: dict) -> dict:
        """Process a new job submission, prepare data, and upsert via VectorService."""
        try:
            # Log the incoming data structure
            logger.info(f"Processing job submission initial data: {json.dumps(job_data, indent=2)}")

            # Validate the job data against the expected input schema (JobPosting)
            try:
                # We still validate against JobPosting to ensure input structure is as expected
                # before flattening. Actual data transformation happens next.
                job_posting = JobPosting(**job_data)
                logger.info("Initial job data validation successful against JobPosting schema")
            except Exception as e:
                logger.error(f"Initial job data validation failed: {str(e)}\n{traceback.format_exc()}")
                raise ValueError(f"Invalid job data format: {str(e)}")

            # Generate an ID if not provided (for upsert matching)
            # The ID column in the DB is likely auto-generated, but upsert needs it if updating
            job_id = job_data.get('id', str(uuid.uuid4()))
            logger.info(f"Using job ID for potential upsert: {job_id}")

            # Prepare flat data dictionary matching the database schema (JobDBModel)
            # This maps from the validated job_data (which might be nested)
            # Convert enums/specific types to strings/ints as needed by the DB
            # Use .get() for safety, although validation should ensure presence
            prepared_flat_data = {
                'id': job_id, # Pass ID for upsert logic
                'job_title': job_data.get('job_title'),
                'job_url': job_data.get('job_url'),
                'positions_available': _safe_int(job_data.get('positions_available'), 1),
                'hiring_urgency': job_data.get('hiring_urgency'),
                'seniority': str(job_data.get('seniority')), # Ensure string if enum
                'work_arrangement': self._ensure_array(job_data.get('work_arrangement')), # Ensure list[str]
                'location_city': self._ensure_array(job_data.get('location_city')),
                'location_state': self._ensure_array(job_data.get('location_state')),
                'location_country': job_data.get('location_country'),
                'visa_sponsorship': job_data.get('visa_sponsorship'),
                'work_authorization': job_data.get('work_authorization'),
                'salary_range_min': _safe_int(job_data.get('salary_range_min')),
                'salary_range_max': _safe_int(job_data.get('salary_range_max')),
                'equity_range_min': str(job_data.get('equity_range_min', '')) if job_data.get('equity_range_min') is not None else None,
                'equity_range_max': str(job_data.get('equity_range_max', '')) if job_data.get('equity_range_max') is not None else None,
                'reporting_structure': job_data.get('reporting_structure'),
                'team_structure': job_data.get('team_structure'),
                'team_roles': self._ensure_array(job_data.get('team_roles')),
                'role_status': str(job_data.get('role_status')), # Ensure string if enum
                'role_category': self._ensure_array(job_data.get('role_category')), # Ensure list[str] if enum
                'tech_stack_must_haves': self._ensure_array(job_data.get('tech_stack_must_haves')),
                'tech_stack_nice_to_haves': self._ensure_array(job_data.get('tech_stack_nice_to_haves')),
                'tech_stack_tags': self._ensure_array(job_data.get('tech_stack_tags')),
                'minimum_years_of_experience': _safe_int(job_data.get('minimum_years_of_experience'), 0),
                'domain_expertise': self._ensure_array(job_data.get('domain_expertise')),
                'ai_ml_exp_required': str(job_data.get('ai_ml_exp_required')), # Ensure string if enum
                'ai_ml_exp_focus': self._ensure_array(job_data.get('ai_ml_exp_focus')),
                'infrastructure_experience': self._ensure_array(job_data.get('infrastructure_experience')),
                'system_design_expectation': job_data.get('system_design_expectation'),
                'coding_proficiency': str(job_data.get('coding_proficiency')), # Ensure string if enum
                'languages': self._ensure_array(job_data.get('languages')),
                'version_control': self._ensure_array(job_data.get('version_control')),
                'ci_cd_tools': self._ensure_array(job_data.get('ci_cd_tools')),
                'collaboration_tools': self._ensure_array(job_data.get('collaboration_tools')),
                'leadership_required': job_data.get('leadership_required'),
                'education_required': str(job_data.get('education_required')), # Ensure string if enum
                'education_advanced_degree': str(job_data.get('education_advanced_degree')), # Ensure string if enum
                'prior_startup_experience': job_data.get('prior_startup_experience'),
                'startup_exp': job_data.get('startup_exp'),
                'career_trajectory': job_data.get('career_trajectory'),
                'independent_work_capacity': job_data.get('independent_work_capacity'),
                'independent_work': job_data.get('independent_work'),
                'skills_must_have': self._ensure_array(job_data.get('skills_must_have')),
                'skills_preferred': self._ensure_array(job_data.get('skills_preferred')),
                'product_description': job_data.get('product_description'),
                'product_stage': str(job_data.get('product_stage')), # Ensure string if enum
                'product_dev_methodology': self._ensure_array(job_data.get('product_dev_methodology')), # Ensure list[str] if enum
                'product_technical_challenges': self._ensure_array(job_data.get('product_technical_challenges')),
                'key_responsibilities': self._ensure_array(job_data.get('key_responsibilities')),
                'scope_of_impact': self._ensure_array(job_data.get('scope_of_impact')),
                'expected_deliverables': self._ensure_array(job_data.get('expected_deliverables')),
                'company_name': job_data.get('company_name'),
                'company_url': job_data.get('company_url'),
                'company_stage': str(job_data.get('company_stage')), # Ensure string if enum
                'company_funding_most_recent': _safe_int(job_data.get('company_funding_most_recent')),
                'company_funding_total': _safe_int(job_data.get('company_funding_total')),
                'company_funding_investors': self._ensure_array(job_data.get('company_funding_investors')),
                'company_founded': str(job_data.get('company_founded', '')) if job_data.get('company_founded') is not None else None,
                'company_team_size': _safe_int(job_data.get('company_team_size'), 0),
                'company_mission': job_data.get('company_mission'),
                'company_vision': job_data.get('company_vision'),
                'company_growth_story': job_data.get('company_growth_story'),
                'company_culture': job_data.get('company_culture'),
                'company_scaling_plans': job_data.get('company_scaling_plans'),
                'company_tech_innovation': job_data.get('company_tech_innovation'),
                'company_industry_vertical': self._ensure_array(job_data.get('company_industry_vertical')), # Ensure list[str] if enum
                'company_target_market': self._ensure_array(job_data.get('company_target_market')), # Ensure list[str] if enum
                'ideal_companies': self._ensure_array(job_data.get('ideal_companies')),
                'deal_breakers': self._ensure_array(job_data.get('deal_breakers')),
                'disqualifying_traits': self._ensure_array(job_data.get('disqualifying_traits')),
                'culture_fit': self._ensure_array(job_data.get('culture_fit')),
                'startup_mindset': self._ensure_array(job_data.get('startup_mindset')),
                'autonomy_level_required': str(job_data.get('autonomy_level_required')), # Ensure string if enum
                'growth_mindset': job_data.get('growth_mindset'),
                'ideal_candidate_profile': job_data.get('ideal_candidate_profile'),
                'interview_process_steps': self._ensure_array(job_data.get('interview_process_steps')),
                'decision_makers': self._ensure_array(job_data.get('decision_makers')),
                'recruiter_pitch_points': self._ensure_array(job_data.get('recruiter_pitch_points')),
            }

            # Remove None values to avoid overwriting existing DB fields with NULL during update
            # The upsert logic in vector_service handles merging
            # prepared_flat_data = {k: v for k, v in prepared_flat_data.items() if v is not None}

            logger.info(f"Prepared flat data for upsert: {json.dumps(prepared_flat_data, indent=2, default=str)}") # Use default=str for potential enums

            # Delegate embedding generation and DB upsert to VectorService
            try:
                # upsert_job handles embedding, metadata, and insert/update
                await self.vector_service.upsert_job(job_id=job_id, job_data=prepared_flat_data)
                logger.info(f"Successfully upserted job {job_id} via VectorService")

                # Return a simplified success response
                return {
                    "status": "success",
                    "message": "Job submission processed and stored successfully.",
                    "job_id": job_id,
                    # Optionally return some basic info if needed, but keep it minimal
                    "details": {
                         "title": prepared_flat_data.get('job_title'),
                         "company": prepared_flat_data.get('company_name')
                    }
                }

            except Exception as e:
                logger.error(f"Error during VectorService upsert for job {job_id}: {str(e)}\n{traceback.format_exc()}")
                # Raise a more specific error or handle as needed
                raise RuntimeError(f"Failed to store job embedding and data: {str(e)}")

        except ValueError as ve: # Catch validation errors
             logger.error(f"Job submission failed validation: {str(ve)}")
             # Return a specific error response for validation failures
             return {
                 "status": "error",
                 "message": f"Job submission failed validation: {str(ve)}",
                 "details": traceback.format_exc()
             }

        except Exception as e:
            logger.error(f"Unexpected error in process_job_submission for job data: {json.dumps(job_data, default=str)}: {str(e)}\n{traceback.format_exc()}")
            # Return a generic error response
            return {
                 "status": "error",
                 "message": "An unexpected error occurred during job processing.",
                 "details": str(e)
             }