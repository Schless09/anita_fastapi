from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
from ..tools.document_processing import PDFProcessor, ResumeParser
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
from ..tools.matching import MatchingTool
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class CandidateIntakeAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory: Optional[Any] = None,
        vector_store: Optional[VectorStoreTool] = None
    ):
        super().__init__(model_name, temperature, memory)
        
        # Initialize tools
        self.tools = [
            PDFProcessor(),
            ResumeParser(),
            vector_store or VectorStoreTool(),  # Use provided vector_store or create a new one
            EmailTool(),
            MatchingTool(vector_store=vector_store)  # Pass vector_store to MatchingTool
        ]
        
        # Initialize agent with system message
        system_message = """You are Anita, an AI recruitment assistant specializing in candidate intake and initial processing.
        Your responsibilities include:
        1. Processing candidate resumes and documents
        2. Extracting relevant information from resumes
        3. Screening candidates based on requirements
        4. Creating candidate profiles in the vector store
        5. Sending confirmation emails to candidates
        6. Handling any errors or issues during the intake process
        
        When screening candidates:
        - Check for required skills and experience
        - Verify work authorization if specified
        - Evaluate location and work arrangement preferences
        - Assess salary expectations against ranges
        - Look for any potential dealbreakers
        
        When responding to users:
        - If a specific question is asked about the candidate intake process, provide a clear, concise answer.
        - If a command is given to process a candidate, acknowledge the request and provide status updates.
        - If a general greeting or unclear message is received, introduce yourself as Anita, explain your role in the recruitment process, and ask how you can assist with candidate intake today.
        - Always include specific, actionable suggestions when possible (e.g., "Would you like to upload a resume?", "Can I help you search for candidates with specific skills?")
        
        Always maintain a professional, friendly tone and ensure data privacy and security when discussing candidate information."""
        
        self._initialize_agent(system_message)

    async def process_candidate(
        self,
        candidate_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a new candidate submission."""
        try:
            # Extract resume data and email if available
            resume_path = candidate_data.get('resume_path')
            resume_content = candidate_data.get('resume_content')
            candidate_email = candidate_data.get('email')
            
            # Initialize response data
            response_data = {
                "status": "processing",
                "candidate_id": candidate_data.get('id'),
                "steps": {}
            }
            
            # Step 1: Process resume if available
            if resume_content:
                logger.info("Processing resume from binary content")
                pdf_tool = PDFProcessor()
                resume_result = await pdf_tool._arun(resume_content)
                response_data["steps"]["resume_processing"] = {
                    "status": "completed",
                    "data": resume_result
                }
                # Update candidate data with resume information
                candidate_data.update(resume_result)
            elif resume_path:
                logger.info(f"Processing resume from file path: {resume_path}")
                pdf_tool = PDFProcessor()
                resume_result = await pdf_tool._arun(resume_path)
                response_data["steps"]["resume_processing"] = {
                    "status": "completed",
                    "data": resume_result
                }
                # Update candidate data with resume information
                candidate_data.update(resume_result)
            else:
                logger.warning("No resume content or path provided")
                response_data["steps"]["resume_processing"] = {
                    "status": "skipped",
                    "reason": "No resume content provided"
                }
            
            # Step 2: Screen candidate
            screening_result = await self.screen_candidate(candidate_data)
            response_data["steps"]["screening"] = {
                "status": "completed",
                "passed": screening_result["passed"],
                "details": screening_result["details"]
            }
            
            # Step 3: Store in vector store if screening passed
            if screening_result["passed"]:
                # Get vector store tool from the tools list
                vector_store = next((tool for tool in self.tools if tool.name == "vector_store"), None)
                if vector_store:
                    logger.info("Storing candidate in vector store")
                    vector_result = await vector_store._arun("store_candidate", candidate_data=candidate_data)
                    response_data["steps"]["vector_store"] = {
                        "status": "completed",
                        "vector_id": vector_result.get("id")
                    }
                else:
                    logger.error("Vector store tool not found")
                    response_data["steps"]["vector_store"] = {
                        "status": "error",
                        "error": "Vector store tool not available"
                    }
                
                # Step 4: Send confirmation email if email available
                if candidate_email:
                    email_tool = next((tool for tool in self.tools if tool.name == "email_tool"), None)
                    if email_tool:
                        logger.info(f"Sending confirmation email to {candidate_email}")
                        email_context = {
                            "candidate_name": candidate_data.get("name", "Candidate"),
                            "screening_result": "passed",
                            "next_steps": screening_result.get("next_steps", [])
                        }
                        email_result = await email_tool._arun(
                            "send_email",
                            to_email=candidate_email,
                            subject="Application Received: Next Steps",
                            content=f"""Dear {candidate_data.get('name', 'Candidate')},

Thank you for submitting your application. We have reviewed your profile and would like to inform you that your application has been processed successfully.

Next Steps:
{screening_result.get('next_steps', ['Our team will review your application in detail.'])}

Best regards,
The Recruitment Team"""
                        )
                        response_data["steps"]["email"] = {
                            "status": "completed",
                            "success": email_result.get("success")
                        }
                    else:
                        logger.error("Email tool not found")
                        response_data["steps"]["email"] = {
                            "status": "error",
                            "error": "Email tool not available"
                        }
            else:
                # Send rejection email if screening failed
                if candidate_email:
                    email_tool = next((tool for tool in self.tools if tool.name == "email_tool"), None)
                    if email_tool:
                        logger.info(f"Sending rejection email to {candidate_email}")
                        email_context = {
                            "candidate_name": candidate_data.get("name", "Candidate"),
                            "screening_result": "not_passed",
                            "reason": screening_result.get("reason", "Did not meet requirements")
                        }
                        email_result = await email_tool._arun(
                            "send_email",
                            to_email=candidate_email,
                            subject="Application Update",
                            content=f"""Dear {candidate_data.get('name', 'Candidate')},

Thank you for your interest in our position. After reviewing your application, we regret to inform you that we will not be moving forward with your candidacy at this time.

Reason: {screening_result.get('reason', 'Your profile does not match our current requirements.')}

We appreciate your time and interest in our company.

Best regards,
The Recruitment Team"""
                        )
                        response_data["steps"]["email"] = {
                            "status": "completed",
                            "success": email_result.get("success")
                        }
                    else:
                        logger.error("Email tool not found")
                        response_data["steps"]["email"] = {
                            "status": "error",
                            "error": "Email tool not available"
                        }
            
            response_data["status"] = "success"
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing candidate: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "steps": response_data.get("steps", {})
            }

    async def screen_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Screen a candidate based on their data."""
        try:
            # Prepare screening criteria
            screening_prompt = f"""Screen this candidate based on their data:
            {candidate_data}
            
            Evaluate:
            1. Skills and experience match
            2. Work authorization (if required)
            3. Location and work arrangement fit
            4. Salary expectations alignment
            5. Potential dealbreakers
            
            Return a structured analysis with:
            - Overall pass/fail decision
            - Detailed reasoning
            - Specific strengths
            - Areas of concern
            - Recommended next steps
            """
            
            screening_result = await self.run(screening_prompt)
            
            # Ensure result is properly structured
            if isinstance(screening_result, str):
                screening_result = eval(screening_result)
            
            return {
                "passed": screening_result.get("passed", False),
                "details": {
                    "reasoning": screening_result.get("reasoning", ""),
                    "strengths": screening_result.get("strengths", []),
                    "concerns": screening_result.get("concerns", []),
                    "next_steps": screening_result.get("next_steps", [])
                },
                "reason": screening_result.get("reason", ""),
                "dealbreakers": screening_result.get("dealbreakers", [])
            }
            
        except Exception as e:
            return {
                "passed": False,
                "details": {
                    "reasoning": f"Error during screening: {str(e)}",
                    "strengths": [],
                    "concerns": ["Error during evaluation"],
                    "next_steps": ["Review manually"]
                },
                "reason": "Technical error during screening",
                "dealbreakers": []
            } 