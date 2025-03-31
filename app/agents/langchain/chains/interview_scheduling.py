from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
from ..tools.matching import MatchingTool

class InterviewSchedulingChain:
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
            self.vector_store = vector_store
            self.matching_tool = MatchingTool(vector_store=vector_store)
        else:
            # This should be avoided!
            self.vector_store = VectorStoreTool()
            self.matching_tool = MatchingTool(vector_store=self.vector_store)
        
        self.email_tool = EmailTool()
        
        # Initialize chains
        self._initialize_chains()

    def _initialize_chains(self):
        """Initialize the scheduling chains."""
        # Interview preparation chain
        interview_prep_prompt = PromptTemplate(
            input_variables=["job_data", "candidate_data"],
            template="""Prepare interview materials for this job-candidate pair:
            
            Job Data:
            {job_data}
            
            Candidate Data:
            {candidate_data}
            
            Provide preparation in JSON format with:
            1. Interview questions
            2. Key topics to cover
            3. Candidate background highlights
            4. Areas to focus on
            5. Red flags to watch for
            6. Success criteria"""
        )
        
        self.interview_prep_chain = interview_prep_prompt | self.llm
        
        # Interview feedback chain
        feedback_analysis_prompt = PromptTemplate(
            input_variables=["feedback", "preparation"],
            template="""Analyze this interview feedback using the preparation materials:
            
            Feedback:
            {feedback}
            
            Preparation:
            {preparation}
            
            Provide analysis in JSON format with:
            1. Overall assessment
            2. Key strengths demonstrated
            3. Areas for improvement
            4. Technical competency
            5. Cultural fit
            6. Next steps recommendation"""
        )
        
        self.feedback_analysis_chain = feedback_analysis_prompt | self.llm

    async def schedule_interview(
        self,
        candidate_id: str,
        job_id: str,
        preferred_times: List[datetime],
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """Schedule an interview between a candidate and hiring manager."""
        try:
            # Step 1: Get candidate and job details
            candidate_result = await self.vector_store._arun(
                "search_candidates",
                query=f"candidate_id:{candidate_id}",
                top_k=1
            )
            
            job_result = await self.vector_store._arun(
                "search_jobs",
                query=f"job_id:{job_id}",
                top_k=1
            )
            
            if candidate_result["status"] != "success" or not candidate_result["results"]:
                return {
                    "status": "error",
                    "error": "Candidate not found"
                }
            
            if job_result["status"] != "success" or not job_result["results"]:
                return {
                    "status": "error",
                    "error": "Job not found"
                }
            
            candidate_data = candidate_result["results"][0]
            job_data = job_result["results"][0]
            
            # Step 2: Prepare interview materials
            prep_result = await self.interview_prep_chain.ainvoke({
                "job_data": job_data["content"],
                "candidate_data": candidate_data["content"]
            })
            
            # Note: Calendar functionality temporarily disabled
            selected_slot = preferred_times[0]
            
            # Step 3: Send confirmation emails
            email_result = await self.email_tool._arun(
                "send_email",
                to_email=candidate_data.get("email"),
                subject=f"Interview Scheduled: {job_data['title']} at {job_data['company']}",
                content=f"""Dear {candidate_data['name']},

Your interview has been scheduled for the {job_data['title']} position at {job_data['company']}.

Interview Details:
- Date: {selected_slot.isoformat()}
- Duration: {duration_minutes} minutes
- Format: Video Call

Please prepare for the following topics:
{prep_result.get('key_topics', [])}

Best regards,
The Recruitment Team"""
            )
            
            return {
                "status": "success",
                "interview_time": selected_slot.isoformat(),
                "email_sent": email_result.get("success"),
                "preparation_materials": prep_result
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
            # Step 1: Get interview details
            interview_result = await self.vector_store._arun(
                "search_interviews",
                query=f"interview_id:{interview_id}",
                top_k=1
            )
            
            if interview_result["status"] != "success" or not interview_result["results"]:
                return {
                    "status": "error",
                    "error": "Interview not found"
                }
            
            interview_data = interview_result["results"][0]
            
            # Step 2: Analyze feedback
            analysis_result = await self.feedback_analysis_chain.ainvoke({
                "feedback": str(feedback),
                "preparation": interview_data.get("preparation_materials", "")
            })
            
            # Step 3: Store feedback and analysis
            store_result = await self.vector_store._arun(
                "store_interview_feedback",
                interview_id=interview_id,
                feedback=feedback,
                analysis=analysis_result
            )
            
            if store_result["status"] != "success":
                return store_result
            
            # Step 4: Send follow-up emails based on analysis
            next_steps = analysis_result.get("next_steps", [])
            if next_steps:
                email_result = await self.email_tool._arun(
                    "send_email",
                    to_email=interview_data.get("candidate_email"),
                    subject="Interview Follow-up",
                    content=f"""Dear {interview_data.get('candidate_name')},

Thank you for your time during the interview. We have reviewed your interview feedback.

Next Steps:
{next_steps}"""
                )
            
            return {
                "status": "success",
                "analysis": analysis_result,
                "next_steps": next_steps
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 