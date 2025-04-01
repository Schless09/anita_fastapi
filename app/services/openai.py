from openai import OpenAI
from app.core.config import settings
from typing import List, Dict, Any

client = OpenAI(api_key=settings.OPENAI_API_KEY)

async def generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings for a given text using OpenAI's API.
    """
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

async def generate_job_embedding(job_data: Dict[str, Any]) -> List[float]:
    """
    Generate embeddings for a job listing by combining relevant fields.
    """
    try:
        # Create a comprehensive text representation of the job
        job_text = f"""
        Title: {job_data['job_title']}
        Company: {job_data['company']['name']}
        Description: {job_data['description']}
        Requirements: {', '.join(job_data['requirements'])}
        Location: {job_data['location']}
        Salary Range: {job_data.get('salary_range', {}).get('min', '')} - {job_data.get('salary_range', {}).get('max', '')}
        Employment Type: {job_data.get('employment_type', '')}
        Role Category: {job_data['role_category']}
        Seniority: {job_data['seniority']}
        Tech Stack: {', '.join(job_data['tech_stack']['must_haves'])}
        Skills Required: {', '.join(job_data['skills_must_have'])}
        Product Description: {job_data['product']['description']}
        Company Mission: {job_data['company']['mission']}
        Key Responsibilities: {', '.join(job_data['key_responsibilities'])}
        Technical Challenges: {', '.join(job_data['product']['technical_challenges'])}
        """
        
        return await generate_embedding(job_text)
        
    except Exception as e:
        print(f"Error generating job embedding: {str(e)}")
        return None 