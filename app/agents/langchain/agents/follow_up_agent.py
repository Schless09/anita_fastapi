from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
from ..tools.matching import MatchingTool
from app.services.vector_service import VectorService
from app.config.settings import Settings

class FollowUpAgent(BaseAgent):
    def __init__(
        self,
        vector_service: VectorService,
        settings: Settings,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory: Optional[Any] = None,
    ):
        super().__init__(model_name, temperature, memory)
        
        # Initialize tools
        vector_store_tool = VectorStoreTool(vector_service=vector_service, settings=settings)
        self.tools = [
            vector_store_tool,
            EmailTool(),
            MatchingTool(vector_store=vector_store_tool)
        ]
        
        # Initialize conversation history
        self.conversation_history = {}
        
        # Initialize agent with system message
        system_message = """You are an AI recruitment assistant specializing in candidate follow-up and engagement.
        Your responsibilities include:
        1. Sending initial job opportunity emails
        2. Analyzing candidate responses
        3. Managing follow-up communications
        4. Tracking conversation history
        5. Providing status updates
        
        When communicating with candidates:
        - Maintain a professional yet friendly tone
        - Be clear about next steps
        - Show enthusiasm for good matches
        - Be respectful of candidate preferences
        - Keep track of conversation context
        
        Always ensure data privacy and maintain professionalism."""
        
        self._initialize_agent(system_message)

    async def contact_candidate(self, job_match: Dict[str, Any]) -> Dict[str, Any]:
        """Send initial job opportunity email to candidate."""
        try:
            # Extract necessary information
            email_context = {
                "recipient_email": job_match.get("email"),
                "job_title": job_match.get("job_details", {}).get("job_title", ""),
                "company_name": job_match.get("job_details", {}).get("company_name", ""),
                "match_score": job_match.get("match_score", 0),
                "match_reason": job_match.get("match_reason", ""),
                "application_link": job_match.get("job_details", {}).get("application_link", "")
            }
            
            # Generate and send email
            result = await self.run(
                f"Send job opportunity email with context: {email_context}"
            )
            
            # Store in conversation history
            self._store_interaction(
                job_match.get("job_id"),
                job_match.get("candidate_id"),
                "initial_contact",
                email_context
            )
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def handle_candidate_reply(
        self,
        candidate_id: str,
        job_id: str,
        email_content: str
    ) -> Dict[str, Any]:
        """Process a candidate's reply to a job opportunity."""
        try:
            # Store the reply
            self._store_interaction(job_id, candidate_id, "candidate_reply", {
                "content": email_content
            })
            
            # Analyze interest
            analysis_prompt = f"""Analyze this candidate reply for interest level:
            {email_content}
            
            Consider:
            1. Explicit statements of interest/disinterest
            2. Questions about the role/company
            3. Concerns or objections raised
            4. Enthusiasm level
            5. Next steps requested
            
            Return a structured analysis with:
            - Interest level (interested/not_interested/needs_more_info)
            - Key points from their response
            - Recommended next steps
            - Any specific questions to address
            """
            
            analysis = await self.run(analysis_prompt)
            
            # Ensure result is properly structured
            if isinstance(analysis, str):
                analysis = eval(analysis)
            
            # Generate and send appropriate follow-up
            follow_up_context = {
                "candidate_id": candidate_id,
                "job_id": job_id,
                "interest_level": analysis.get("interest_level"),
                "key_points": analysis.get("key_points", []),
                "next_steps": analysis.get("next_steps", []),
                "questions": analysis.get("questions", [])
            }
            
            result = await self.run(
                f"Send follow-up email with context: {follow_up_context}"
            )
            
            # Store the follow-up
            self._store_interaction(job_id, candidate_id, "follow_up", follow_up_context)
            
            return {
                "status": "success",
                "analysis": analysis,
                "email_sent": result.get("success", False)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def send_missed_call_email(
        self,
        candidate_id: str,
        email: str,
        first_name: str = ""
    ) -> Dict[str, Any]:
        """Send an email when a scheduled call was missed."""
        try:
            email_context = {
                "candidate_name": first_name or "there",
                "email": email,
                "reschedule_link": "YOUR_SCHEDULING_LINK"  # Replace with actual link
            }
            
            result = await self.run(
                f"Send missed call email with context: {email_context}"
            )
            
            # Store the interaction
            self._store_interaction(None, candidate_id, "missed_call", email_context)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def send_transcript_summary(
        self,
        candidate_id: str,
        email: str,
        processed_data: Dict[str, Any],
        call_status: str = "completed"
    ) -> Dict[str, Any]:
        """Send a summary email after a call with transcript highlights."""
        try:
            email_context = {
                "email": email,
                "call_status": call_status,
                "highlights": processed_data.get("highlights", []),
                "next_steps": processed_data.get("next_steps", []),
                "action_items": processed_data.get("action_items", [])
            }
            
            result = await self.run(
                f"Send transcript summary email with context: {email_context}"
            )
            
            # Store the interaction
            self._store_interaction(None, candidate_id, "transcript_summary", email_context)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _store_interaction(
        self,
        job_id: Optional[str],
        candidate_id: str,
        interaction_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Store an interaction in the conversation history."""
        if job_id:
            if job_id not in self.conversation_history:
                self.conversation_history[job_id] = {
                    "candidate_id": candidate_id,
                    "interactions": []
                }
            
            self.conversation_history[job_id]["interactions"].append({
                "type": interaction_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            })
        else:
            # Store by candidate_id if no job_id (e.g., for calls)
            if candidate_id not in self.conversation_history:
                self.conversation_history[candidate_id] = {
                    "interactions": []
                }
            
            self.conversation_history[candidate_id]["interactions"].append({
                "type": interaction_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            })

    async def check_pending_follow_ups(
        self,
        days_threshold: int = 7
    ) -> Dict[str, Any]:
        """Check for candidates needing follow-up."""
        try:
            # Get candidates needing follow-up
            candidates = await self.run(
                f"Find candidates who haven't responded in {days_threshold} days"
            )
            
            return {
                "status": "success",
                "candidates_needing_follow_up": candidates
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def send_follow_up(
        self,
        candidate_id: str,
        context: Dict[str, Any],
        follow_up_type: str
    ) -> Dict[str, Any]:
        """Send a follow-up message to a candidate."""
        try:
            # Get candidate details
            candidate_data = await self.run(f"Get candidate details for candidate_id: {candidate_id}")
            
            # Generate personalized follow-up message
            message = await self.run(
                f"Generate personalized follow-up message for {follow_up_type} "
                f"with context: {context}"
            )
            
            # Send follow-up email
            email_result = await self.run(
                f"Send follow-up email to {candidate_data['email']} with message: {message}"
            )
            
            # Update follow-up status
            status_result = await self.run(
                f"Update follow-up status for candidate {candidate_id}"
            )
            
            return {
                "status": "success",
                "email_sent": email_result.get("success"),
                "follow_up_status": status_result.get("status")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def handle_candidate_response(
        self,
        candidate_id: str,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process candidate's response to follow-up."""
        try:
            # Analyze response
            analysis = await self.run(
                f"Analyze candidate response: {response}"
            )
            
            # Update candidate status
            status_result = await self.run(
                f"Update candidate status based on response analysis: {analysis}"
            )
            
            # Determine next actions
            next_actions = await self.run(
                f"Determine next actions based on response: {analysis}"
            )
            
            return {
                "status": "success",
                "analysis": analysis,
                "status_updated": status_result.get("success"),
                "next_actions": next_actions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def generate_follow_up_report(
        self,
        time_period: timedelta
    ) -> Dict[str, Any]:
        """Generate a report of follow-up activities."""
        try:
            # Get follow-up statistics
            stats = await self.run(
                f"Get follow-up statistics for the last {time_period}"
            )
            
            # Generate report
            report = await self.run(
                f"Generate follow-up report with statistics: {stats}"
            )
            
            return {
                "status": "success",
                "statistics": stats,
                "report": report
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 