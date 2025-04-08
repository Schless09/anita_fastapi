from typing import Dict, Any, Optional, List
import logging
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.config.supabase import get_supabase_client
from supabase._async.client import AsyncClient
from app.config.settings import get_settings, Settings
from app.config.utils import get_table_name
import uuid

logger = logging.getLogger(__name__)

class MatchingService:
    def __init__(self, 
                 openai_service: OpenAIService, 
                 vector_service: VectorService, 
                 supabase_client: AsyncClient, 
                 settings: Settings):
        self.openai_service = openai_service
        self.vector_service = vector_service
        self.supabase: AsyncClient = supabase_client
        self.candidates_table_name = get_table_name("candidates", settings)
    
    async def match_candidate_to_jobs(self, candidate_id: uuid.UUID, top_k: int = 10) -> List[Dict[str, Any]]:
        """Match a candidate to potential jobs based on their stored embedding."""
        logger.info(f"Starting job matching for candidate: {candidate_id}")
        try:
            # 1. Fetch candidate's embedding from Supabase
            response = await self.supabase.table(self.candidates_table_name).select('embedding').eq('id', str(candidate_id)).maybe_single().execute()
            
            if not response.data or not response.data.get('embedding'):
                logger.warning(f"No embedding found for candidate {candidate_id}. Cannot perform matching.")
                return []
            
            embedding = response.data['embedding']
            
            # 2. Query vector database for matching jobs using the fetched embedding
            matches = await self.vector_service.query_jobs(embedding, top_k)
            
            # 3. Format and return results
            return [
                {
                    "job_id": job.get("id"),
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "similarity": job.get("similarity", 0),
                    "job_data": job.get("profile_json", {})
                }
                for job in matches
            ]
        
        except Exception as e:
            logger.error(f"Error matching candidate {candidate_id} to jobs: {str(e)}")
            raise
    
    async def match_job_to_candidates(self, job_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Match a job to potential candidates based on its description."""
        try:
            # 1. Generate embedding for job text
            job_text = self._prepare_job_for_matching(job_data)
            embedding = await self.openai_service.generate_embedding(job_text)
            
            # 2. Query vector database for matching candidates
            matches = await self.vector_service.query_candidates(embedding, top_k)
            
            # 3. Format and return results
            return [
                {
                    "candidate_id": candidate.get("id"),
                    "name": candidate.get("full_name"),
                    "email": candidate.get("email"),
                    "similarity": candidate.get("similarity", 0),
                    "candidate_data": candidate.get("profile_json", {})
                }
                for candidate in matches
            ]
        
        except Exception as e:
            logger.error(f"Error matching job to candidates: {str(e)}")
            raise
    
    def _prepare_job_for_matching(self, job_data: Dict[str, Any]) -> str:
        """Format job data for matching using all available job information."""
        sections = []
        
        # Company Information
        sections.append("COMPANY INFORMATION:")
        sections.append(f"Company Name: {job_data.get('company_name', '')}")
        sections.append(f"Company URL: {job_data.get('company_url', '')}")
        sections.append(f"Company Stage: {job_data.get('company_stage', '')}")
        sections.append(f"Company Mission: {job_data.get('company_mission', '')}")
        sections.append(f"Company Vision: {job_data.get('company_vision', '')}")
        sections.append(f"Company Culture: {job_data.get('company_culture', '')}")
        sections.append(f"Company Growth Story: {job_data.get('company_growth_story', '')}")
        sections.append(f"Company Scaling Plans: {job_data.get('company_scaling_plans', '')}")
        sections.append(f"Company Tech Innovation: {job_data.get('company_tech_innovation', '')}")
        sections.append(f"Company Team Size: {job_data.get('company_team_size', '')}")
        sections.append(f"Company Founded: {job_data.get('company_founded', '')}")
        
        # Company Industry and Market
        sections.append("\nINDUSTRY AND MARKET:")
        sections.append(f"Industry Verticals: {', '.join(job_data.get('company_industry_vertical', []))}")
        sections.append(f"Target Markets: {', '.join(job_data.get('company_target_market', []))}")
        
        # Job Details
        sections.append("\nJOB DETAILS:")
        sections.append(f"Job Title: {job_data.get('job_title', '')}")
        sections.append(f"Job URL: {job_data.get('job_url', '')}")
        sections.append(f"Role Category: {', '.join(job_data.get('role_category', []))}")
        sections.append(f"Seniority: {job_data.get('seniority', '')}")
        sections.append(f"Positions Available: {job_data.get('positions_available', '')}")
        sections.append(f"Role Status: {job_data.get('role_status', '')}")
        
        # Location and Work Arrangement
        sections.append("\nLOCATION AND WORK ARRANGEMENT:")
        sections.append(f"Location Country: {job_data.get('location_country', '')}")
        sections.append(f"Location State: {', '.join(job_data.get('location_state', []))}")
        sections.append(f"Location City: {', '.join(job_data.get('location_city', []))}")
        sections.append(f"Work Arrangement: {', '.join(job_data.get('work_arrangement', []))}")
        sections.append(f"Work Authorization: {job_data.get('work_authorization', '')}")
        sections.append(f"Visa Sponsorship: {'Yes' if job_data.get('visa_sponsorship') else 'No'}")
        
        # Technical Requirements
        sections.append("\nTECHNICAL REQUIREMENTS:")
        sections.append(f"Tech Stack (Must Have): {', '.join(job_data.get('tech_stack_must_haves', []))}")
        sections.append(f"Tech Stack (Nice to Have): {', '.join(job_data.get('tech_stack_nice_to_haves', []))}")
        sections.append(f"Tech Stack Tags: {', '.join(job_data.get('tech_stack_tags', []))}")
        sections.append(f"Languages: {', '.join(job_data.get('languages', []))}")
        sections.append(f"CI/CD Tools: {', '.join(job_data.get('ci_cd_tools', []))}")
        sections.append(f"Version Control: {', '.join(job_data.get('version_control', []))}")
        sections.append(f"Infrastructure Experience: {', '.join(job_data.get('infrastructure_experience', []))}")
        sections.append(f"Product Development Methodology: {', '.join(job_data.get('product_dev_methodology', []))}")
        sections.append(f"System Design Expectation: {job_data.get('system_design_expectation', '')}")
        sections.append(f"Coding Proficiency: {job_data.get('coding_proficiency', '')}")
        
        # Skills and Experience
        sections.append("\nSKILLS AND EXPERIENCE:")
        sections.append(f"Skills (Must Have): {', '.join(job_data.get('skills_must_have', []))}")
        sections.append(f"Skills (Preferred): {', '.join(job_data.get('skills_preferred', []))}")
        sections.append(f"Domain Expertise: {', '.join(job_data.get('domain_expertise', []))}")
        sections.append(f"AI/ML Experience Required: {job_data.get('ai_ml_exp_required', '')}")
        sections.append(f"AI/ML Experience Focus: {', '.join(job_data.get('ai_ml_exp_focus', []))}")
        sections.append(f"Minimum Years of Experience: {job_data.get('minimum_years_of_experience', '')}")
        sections.append(f"Prior Startup Experience Required: {'Yes' if job_data.get('prior_startup_experience') else 'No'}")
        sections.append(f"Startup Experience: {job_data.get('startup_exp', '')}")
        
        # Education and Qualifications
        sections.append("\nEDUCATION AND QUALIFICATIONS:")
        sections.append(f"Education Required: {job_data.get('education_required', '')}")
        sections.append(f"Advanced Degree Required: {job_data.get('education_advanced_degree', '')}")
        sections.append(f"Leadership Required: {'Yes' if job_data.get('leadership_required') else 'No'}")
        sections.append(f"Growth Mindset: {job_data.get('growth_mindset', '')}")
        sections.append(f"Autonomy Level Required: {job_data.get('autonomy_level_required', '')}")
        sections.append(f"Independent Work: {job_data.get('independent_work', '')}")
        sections.append(f"Independent Work Capacity: {job_data.get('independent_work_capacity', '')}")
        
        # Product and Team
        sections.append("\nPRODUCT AND TEAM:")
        sections.append(f"Product Description: {job_data.get('product_description', '')}")
        sections.append(f"Product Stage: {job_data.get('product_stage', '')}")
        sections.append(f"Product Technical Challenges: {job_data.get('product_technical_challenges', '')}")
        sections.append(f"Team Structure: {job_data.get('team_structure', '')}")
        sections.append(f"Team Roles: {', '.join(job_data.get('team_roles', []))}")
        sections.append(f"Reporting Structure: {job_data.get('reporting_structure', '')}")
        
        # Responsibilities and Impact
        sections.append("\nRESPONSIBILITIES AND IMPACT:")
        sections.append(f"Key Responsibilities: {', '.join(job_data.get('key_responsibilities', []))}")
        sections.append(f"Expected Deliverables: {', '.join(job_data.get('expected_deliverables', []))}")
        sections.append(f"Scope of Impact: {', '.join(job_data.get('scope_of_impact', []))}")
        
        # Culture and Fit
        sections.append("\nCULTURE AND FIT:")
        sections.append(f"Culture Fit: {', '.join(job_data.get('culture_fit', []))}")
        sections.append(f"Startup Mindset: {', '.join(job_data.get('startup_mindset', []))}")
        sections.append(f"Deal Breakers: {', '.join(job_data.get('deal_breakers', []))}")
        sections.append(f"Disqualifying Traits: {', '.join(job_data.get('disqualifying_traits', []))}")
        sections.append(f"Ideal Candidate Profile: {job_data.get('ideal_candidate_profile', '')}")
        sections.append(f"Ideal Companies: {', '.join(job_data.get('ideal_companies', []))}")
        
        # Compensation and Benefits
        sections.append("\nCOMPENSATION AND BENEFITS:")
        sections.append(f"Salary Range: ${job_data.get('salary_range_min', '')} - ${job_data.get('salary_range_max', '')}")
        sections.append(f"Equity Range: {job_data.get('equity_range_min', '')} - {job_data.get('equity_range_max', '')}")
        
        # Hiring Process
        sections.append("\nHIRING PROCESS:")
        sections.append(f"Hiring Urgency: {job_data.get('hiring_urgency', '')}")
        sections.append(f"Interview Process Steps: {', '.join(job_data.get('interview_process_steps', []))}")
        sections.append(f"Decision Makers: {', '.join(job_data.get('decision_makers', []))}")
        sections.append(f"Recruiter Pitch Points: {', '.join(job_data.get('recruiter_pitch_points', []))}")
        
        return "\n".join(sections) 