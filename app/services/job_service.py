from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
from app.config import get_settings
from app.config.supabase import get_supabase_client
from app.services.openai_service import OpenAIService
from app.services.pinecone_service import PineconeService

settings = get_settings()
supabase = get_supabase_client()
openai = OpenAIService()
pinecone = PineconeService()

class JobService:
    def __init__(self):
        self.supabase = supabase
        self.pinecone_service = PineconeService()

    async def process_job_submission(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new job submission.
        """
        try:
            # 1. Create company first
            company_data = self._extract_company_data(job_data)
            company_id = str(uuid.uuid4())
            company_data['id'] = company_id
            company_data['created_at'] = datetime.utcnow().isoformat()
            company_data['updated_at'] = datetime.utcnow().isoformat()

            company_response = await self.supabase.table('companies_dev').insert(company_data).execute()
            created_company = company_response.data[0]

            # 2. Create job linked to company
            job_id = str(uuid.uuid4())
            job_data['id'] = job_id
            job_data['company_id'] = company_id
            job_data['created_at'] = datetime.utcnow().isoformat()
            job_data['updated_at'] = datetime.utcnow().isoformat()

            # Remove company data from job_data
            job_data = self._clean_job_data(job_data)

            job_response = await self.supabase.table('jobs_dev').insert(job_data).execute()
            created_job = job_response.data[0]

            # 3. Send combined data to Pinecone
            combined_profile = self._combine_job_company_data(created_job, created_company)
            pinecone_id = await self.pinecone_service.upsert_job(
                job_id=job_id,
                profile_data=combined_profile
            )

            # 4. Update job with Pinecone ID
            await self.supabase.table('jobs_dev').update({
                'pinecone_id': pinecone_id,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', job_id).execute()

            return {
                'job_id': job_id,
                'company_id': company_id,
                'status': 'created'
            }

        except Exception as e:
            raise Exception(f"Error processing job submission: {str(e)}")

    def _extract_company_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract company data from job submission.
        """
        company_fields = {
            'name': job_data.pop('company_name', ''),
            'url': job_data.pop('company_url', ''),
            'stage': job_data.pop('company_stage', ''),
            'team_size': job_data.pop('company_size', ''),
            'founded': job_data.pop('company_founded', ''),
            'mission': job_data.pop('company_mission', ''),
            'vision': job_data.pop('company_vision', ''),
            'mission_and_impact': job_data.pop('company_mission_and_impact', ''),
            'growth_story': job_data.pop('company_growth_story', ''),
            'culture': job_data.pop('company_culture', ''),
            'scaling_plans': job_data.pop('company_scaling_plans', ''),
            'tech_innovation': job_data.pop('company_tech_innovation', ''),
            'funding_most_recent_round': job_data.pop('company_funding_round', ''),
            'funding_total': job_data.pop('company_funding_total', ''),
            'funding_investors': job_data.pop('company_investors', [])
        }
        return {k: v for k, v in company_fields.items() if v}

    def _clean_job_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove any remaining company fields from job data.
        """
        company_prefixed_fields = [
            field for field in job_data.keys()
            if field.startswith('company_')
        ]
        for field in company_prefixed_fields:
            job_data.pop(field, None)
        return job_data

    def _combine_job_company_data(self, job: Dict[str, Any], company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine job and company data for Pinecone storage.
        """
        return {
            'job_id': job['id'],
            'company_id': company['id'],
            'job_title': job.get('job_title', ''),
            'company_name': company.get('name', ''),
            'company_stage': company.get('stage', ''),
            'company_size': company.get('team_size', ''),
            'tech_stack': job.get('tech_stack_must_haves', []) + job.get('tech_stack_nice_to_haves', []),
            'salary_range': {
                'min': job.get('salary_min', 0),
                'max': job.get('salary_max', 0)
            },
            'location': {
                'city': job.get('city', []),
                'state': job.get('state', []),
                'country': job.get('country', '')
            },
            'work_arrangement': job.get('work_arrangement', []),
            'company_mission': company.get('mission', ''),
            'company_culture': company.get('culture', ''),
            'funding_stage': company.get('funding_most_recent_round', ''),
            'domain_expertise': job.get('domain_expertise', []),
            'seniority_level': job.get('seniority_level', ''),
            'minimum_years_of_experience': job.get('minimum_years_of_experience', '')
        } 