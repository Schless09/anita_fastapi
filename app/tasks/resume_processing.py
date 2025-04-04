from typing import Any, Dict
import asyncio
import logging
from app.agents.langchain.agents.candidate_intake_agent import CandidateIntakeAgent
from app.services.vector_service import VectorService
from app.config.settings import Settings
from app.config.supabase import get_supabase_client

logger = logging.getLogger(__name__)

class ResumeProcessingTask:
    def __init__(
        self,
        vector_service: VectorService,
        settings: Settings,
        supabase=None
    ):
        self.vector_service = vector_service
        self.settings = settings
        self.supabase = supabase or get_supabase_client()
        
    async def process_resume(
        self,
        resume_content: bytes,
        candidate_id: str
    ) -> None:
        """Process a resume in the background."""
        try:
            # Create intake agent
            agent = CandidateIntakeAgent(
                vector_service=self.vector_service,
                settings=self.settings,
                candidate_id=candidate_id,
                supabase=self.supabase
            )
            
            # Process the resume
            await agent._process_full_resume_background(
                resume_content=resume_content,
                candidate_id=candidate_id
            )
            
        except Exception as e:
            logger.error(f"Error in background resume processing task: {str(e)}")
            # Update candidate status to error
            await self.supabase.table("candidates").update({
                "status": "error",
                "error_message": str(e)
            }).eq("id", candidate_id).execute()

async def start_resume_processing(
    resume_content: bytes,
    candidate_id: str,
    vector_service: VectorService,
    settings: Settings
) -> None:
    """Start the resume processing task in the background."""
    try:
        task = ResumeProcessingTask(
            vector_service=vector_service,
            settings=settings
        )
        # Start the task in the background
        asyncio.create_task(task.process_resume(resume_content, candidate_id))
    except Exception as e:
        logger.error(f"Error starting resume processing task: {str(e)}")
        raise 