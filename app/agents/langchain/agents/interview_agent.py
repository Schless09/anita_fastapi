from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
from ..tools.matching import MatchingTool

class InterviewAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ):
        super().__init__(model_name, temperature)
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize tools
        self.tools = [
            EmailTool(),
            VectorStoreTool(),
            MatchingTool()
        ]
        
        # Initialize agent with system message
        system_message = """You are an AI recruitment assistant specializing in interview coordination.
        Your responsibilities include:
        1. Scheduling interviews between candidates and hiring managers
        2. Managing calendar availability and time zones
        3. Sending interview invitations and confirmations
        4. Preparing interview materials and questions
        5. Handling interview feedback and next steps
        
        Always be professional and considerate of everyone's time."""
        
        self._initialize_agent(system_message)

    async def schedule_interview(
        self,
        candidate_id: str,
        job_id: str,
        preferred_times: List[datetime],
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """Schedule an interview between a candidate and hiring manager."""
        try:
            # Get candidate and job details
            candidate_data = await self.run(f"Get candidate details for candidate_id: {candidate_id}")
            job_data = await self.run(f"Get job details for job_id: {job_id}")
            
            # Find available time slots
            available_slots = await self.run(
                f"Find available time slots for {duration_minutes} minutes "
                f"among preferred times: {preferred_times}"
            )
            
            if not available_slots:
                return {
                    "status": "error",
                    "error": "No available time slots found"
                }
            
            # Schedule the interview
            schedule_result = await self.run(
                f"Schedule interview for candidate {candidate_id} "
                f"with job {job_id} at time: {available_slots[0]}"
            )
            
            # Send calendar invites
            calendar_result = await self.run(
                f"Send calendar invites for interview at {available_slots[0]}"
            )
            
            # Send confirmation emails
            email_result = await self.run(
                f"Send interview confirmation emails for {available_slots[0]}"
            )
            
            return {
                "status": "success",
                "interview_time": available_slots[0],
                "calendar_event_id": calendar_result.get("event_id"),
                "emails_sent": email_result.get("success")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def prepare_interview(
        self,
        interview_id: str
    ) -> Dict[str, Any]:
        """Prepare interview materials and questions."""
        try:
            # Get interview details
            interview_data = await self.run(f"Get interview details for interview_id: {interview_id}")
            
            # Generate interview questions
            questions = await self.run(
                f"Generate interview questions for job: {interview_data['job_id']}"
            )
            
            # Prepare candidate background
            background = await self.run(
                f"Prepare candidate background for: {interview_data['candidate_id']}"
            )
            
            return {
                "status": "success",
                "questions": questions,
                "candidate_background": background
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def handle_interview_feedback(
        self,
        interview_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process interview feedback and determine next steps."""
        try:
            # Store feedback
            feedback_result = await self.run(
                f"Store interview feedback for interview_id: {interview_id}"
            )
            
            # Analyze feedback
            analysis = await self.run(
                f"Analyze interview feedback: {feedback}"
            )
            
            # Determine next steps
            next_steps = await self.run(
                f"Determine next steps based on feedback analysis: {analysis}"
            )
            
            return {
                "status": "success",
                "feedback_stored": feedback_result.get("success"),
                "analysis": analysis,
                "next_steps": next_steps
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 