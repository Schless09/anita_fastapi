from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
from ..tools.matching import MatchingTool
import logging

logger = logging.getLogger(__name__)

class FollowUpChain:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        vector_store: Optional[VectorStoreTool] = None
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize tools
        if vector_store:
            logger.info("FollowUpChain using provided VectorStoreTool instance")
            self.vector_store = vector_store
        else:
            logger.warning("⚠️ FollowUpChain creating new VectorStoreTool - this should be avoided!")
            self.vector_store = VectorStoreTool()
            
        self.email_tool = EmailTool()
        self.matching_tool = MatchingTool(vector_store=self.vector_store)
        
        # Initialize chains
        self._initialize_chains()

    def _initialize_chains(self):
        """Initialize the follow-up chains."""
        # Response analysis chain
        response_analysis_prompt = PromptTemplate(
            input_variables=["response", "context"],
            template="""Analyze this candidate's response to our follow-up:
            
            Response:
            {response}
            
            Context:
            {context}
            
            Provide analysis in JSON format with:
            1. Response sentiment
            2. Key points addressed
            3. Questions or concerns
            4. Interest level
            5. Next steps needed
            6. Follow-up recommendations"""
        )
        
        self.response_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=response_analysis_prompt
        )
        
        # Follow-up generation chain
        follow_up_generation_prompt = PromptTemplate(
            input_variables=["analysis", "history"],
            template="""Generate a follow-up message based on this analysis and history:
            
            Analysis:
            {analysis}
            
            History:
            {history}
            
            Generate a personalized message in JSON format with:
            1. Subject line
            2. Opening greeting
            3. Main message
            4. Specific questions or requests
            5. Closing statement
            6. Next steps"""
        )
        
        self.follow_up_generation_chain = LLMChain(
            llm=self.llm,
            prompt=follow_up_generation_prompt
        )

    async def check_pending_follow_ups(
        self,
        days_threshold: int = 7
    ) -> Dict[str, Any]:
        """Check for candidates needing follow-up."""
        try:
            # Get candidates needing follow-up
            candidates_result = await self.vector_store._arun(
                "search_candidates",
                query=f"last_contact:{days_threshold}",
                top_k=10
            )
            
            if candidates_result["status"] != "success":
                return candidates_result
            
            # Filter candidates based on status
            candidates_needing_follow_up = []
            for candidate in candidates_result["results"]:
                if self._needs_follow_up(candidate):
                    candidates_needing_follow_up.append(candidate)
            
            return {
                "status": "success",
                "candidates_needing_follow_up": candidates_needing_follow_up
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
            # Step 1: Get candidate details
            candidate_result = await self.vector_store._arun(
                "search_candidates",
                query=f"candidate_id:{candidate_id}",
                top_k=1
            )
            
            if candidate_result["status"] != "success" or not candidate_result["results"]:
                return {
                    "status": "error",
                    "error": "Candidate not found"
                }
            
            candidate_data = candidate_result["results"][0]
            
            # Step 2: Get communication history
            history_result = await self.vector_store._arun(
                "search_communications",
                query=f"candidate_id:{candidate_id}",
                top_k=5
            )
            
            # Step 3: Generate follow-up message
            follow_up_result = await self.follow_up_generation_chain.arun(
                analysis=str(context),
                history=str(history_result.get("results", []))
            )
            
            # Step 4: Send follow-up email
            email_result = await self.email_tool._arun(
                "send_email",
                to_email=candidate_data.get("email"),
                subject=follow_up_result.get("subject", "Follow-up Message"),
                content=follow_up_result.get("message", "")
            )
            
            # Step 5: Store communication record
            store_result = await self.vector_store._arun(
                "store_communication",
                candidate_id=candidate_id,
                type=follow_up_type,
                content=follow_up_result.get("message", ""),
                metadata={
                    "subject": follow_up_result.get("subject"),
                    "next_steps": follow_up_result.get("next_steps")
                }
            )
            
            return {
                "status": "success",
                "email_sent": email_result.get("success"),
                "communication_stored": store_result.get("success"),
                "next_steps": follow_up_result.get("next_steps")
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
            # Step 1: Get candidate details and context
            candidate_result = await self.vector_store._arun(
                "search_candidates",
                query=f"candidate_id:{candidate_id}",
                top_k=1
            )
            
            if candidate_result["status"] != "success" or not candidate_result["results"]:
                return {
                    "status": "error",
                    "error": "Candidate not found"
                }
            
            candidate_data = candidate_result["results"][0]
            
            # Step 2: Analyze response
            analysis_result = await self.response_analysis_chain.arun(
                response=str(response),
                context=str(candidate_data)
            )
            
            # Step 3: Update candidate status
            update_result = await self.vector_store._arun(
                "update_candidate_status",
                candidate_id=candidate_id,
                status=analysis_result.get("status"),
                metadata=analysis_result
            )
            
            # Step 4: Determine next actions
            next_actions = analysis_result.get("next_steps", [])
            
            # Step 5: Send acknowledgment if needed
            if next_actions:
                email_result = await self.email_tool._arun(
                    "send_email",
                    to_email=candidate_data.get("email"),
                    subject="Thank You for Your Response",
                    content=f"""Dear {candidate_data.get('name')},

Thank you for your response. We have noted your feedback and will be in touch soon with next steps.

Best regards,
The Recruitment Team"""
                )
            
            return {
                "status": "success",
                "analysis": analysis_result,
                "status_updated": update_result.get("success"),
                "next_actions": next_actions,
                "acknowledgment_sent": email_result.get("success") if next_actions else False
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
            # Step 1: Get follow-up statistics
            stats_result = await self.vector_store._arun(
                "get_follow_up_stats",
                time_period=time_period
            )
            
            if stats_result["status"] != "success":
                return stats_result
            
            # Step 2: Generate report
            report_prompt = f"""Generate a follow-up activity report using these statistics:
            
            Statistics:
            {stats_result['stats']}
            
            Provide a report in JSON format with:
            1. Summary of activities
            2. Response rates
            3. Key trends
            4. Areas for improvement
            5. Recommendations"""
            
            report_result = await self.llm.ainvoke(report_prompt)
            
            return {
                "status": "success",
                "statistics": stats_result["stats"],
                "report": report_result.content
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _needs_follow_up(self, candidate: Dict[str, Any]) -> bool:
        """Determine if a candidate needs follow-up."""
        try:
            # Check last contact date
            last_contact = candidate.get("last_contact")
            if not last_contact:
                return True
            
            last_contact_date = datetime.fromisoformat(last_contact)
            days_since_contact = (datetime.now() - last_contact_date).days
            
            # Check candidate status
            status = candidate.get("status")
            
            # Define follow-up rules
            if status == "new":
                return days_since_contact >= 3
            elif status == "in_progress":
                return days_since_contact >= 7
            elif status == "interviewed":
                return days_since_contact >= 5
            else:
                return False
                
        except Exception:
            return False 