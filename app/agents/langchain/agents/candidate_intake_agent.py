from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
from ..tools.document_processing import PDFProcessor, ResumeParser
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
from ..tools.matching import MatchingTool
from pydantic import BaseModel
import logging
import traceback
from langchain.prompts import PromptTemplate
from datetime import datetime
from app.config import get_supabase_client

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
        
        # Initialize tools as instance attributes
        self.pdf_processor = PDFProcessor()
        self.resume_parser = ResumeParser()
        self.vector_store = vector_store or VectorStoreTool()
        self.email_tool = EmailTool()
        self.matching_tool = MatchingTool(vector_store=vector_store)
        
        # Initialize Supabase client
        self.supabase = get_supabase_client()
        
        # Set tools list for the agent
        self.tools = [
            self.pdf_processor,
            self.resume_parser,
            self.vector_store,
            self.email_tool,
            self.matching_tool
        ]
        
        # Initialize chains
        self._initialize_chains()
        
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

    def _initialize_chains(self):
        """Initialize the processing chains."""
        # Resume analysis chain
        resume_analysis_prompt = PromptTemplate(
            input_variables=["resume_text"],
            template="""Analyze this resume and provide a detailed assessment:
            
            Resume:
            {resume_text}
            
            Provide analysis in JSON format with:
            1. Overall assessment
            2. Key skills
            3. Experience level
            4. Education
            5. Notable achievements
            6. Areas for improvement"""
        )
        
        self.resume_analysis_chain = resume_analysis_prompt | self.llm
        
        # Profile creation chain
        profile_creation_prompt = PromptTemplate(
            input_variables=["parsed_data", "analysis"],
            template="""Create a comprehensive candidate profile using the parsed data and analysis.
            You MUST return a valid JSON object with the following structure:
            {{
                "basic_info": {{
                    "name": "string",
                    "email": "string",
                    "phone": "string",
                    "location": "string"
                }},
                "professional_summary": "string",
                "skills": ["string"],
                "experience": [
                    {{
                        "title": "string",
                        "company": "string",
                        "duration": "string",
                        "description": "string"
                    }}
                ],
                "education": [
                    {{
                        "degree": "string",
                        "institution": "string",
                        "year": "string"
                    }}
                ],
                "additional_qualifications": ["string"]
            }}
            
            Use the following data to create the profile:
            
            Parsed Data:
            {parsed_data}
            
            Analysis:
            {analysis}
            
            Return ONLY the JSON object, no other text."""
        )
        
        self.profile_creation_chain = profile_creation_prompt | self.llm

    async def process_candidate(
        self,
        resume_content: bytes,
        candidate_email: str,
        candidate_id: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            logger.info("\n=== Starting Candidate Intake Process ===")
            logger.info(f"Processing candidate: {candidate_email}")
            
            # Step 1: Parse Resume
            logger.info("\nStep 1: 📄 Parsing Resume")
            logger.info("----------------------------------------")
            
            # First convert PDF to text
            pdf_result = await self.pdf_processor._arun(resume_content)
            if pdf_result["status"] != "success":
                logger.error(f"❌ PDF Processing Failed: {pdf_result}")
                return pdf_result
                
            # Then parse the text content
            parse_result = await self.resume_parser._arun(pdf_result["text_content"])
            
            if parse_result["status"] != "success":
                logger.error(f"❌ Resume Parsing Failed: {parse_result}")
                return parse_result
            logger.info("✅ Resume parsed successfully")
            
            # Step 2: Analyze Resume
            logger.info("\nStep 2: 🔍 Analyzing Resume")
            logger.info("----------------------------------------")
            analysis_result = await self.resume_analysis_chain.ainvoke(
                {"resume_text": pdf_result["text_content"]}
            )
            
            if not analysis_result:
                logger.error("❌ Resume Analysis Failed")
                return {"status": "error", "error": "Failed to analyze resume"}
            logger.info("✅ Resume analyzed successfully")
            
            # Step 3: Create Profile
            logger.info("\nStep 3: 👤 Creating Profile")
            logger.info("----------------------------------------")
            profile_result = await self.profile_creation_chain.ainvoke(
                {"parsed_data": parse_result["parsed_data"], "analysis": analysis_result}
            )
            logger.info("✅ Profile created successfully")
            
            # Step 4: Store in Supabase
            logger.info("\nStep 4: 💾 Storing in Supabase")
            logger.info("----------------------------------------")
            
            # Convert AIMessage to string if needed and parse JSON
            if hasattr(profile_result, 'content'):
                profile_result = profile_result.content
            
            try:
                # Try to parse the profile result as JSON
                if isinstance(profile_result, str):
                    import json
                    # Clean the string to ensure it's valid JSON
                    profile_result = profile_result.strip()
                    if profile_result.startswith('```json'):
                        profile_result = profile_result[7:]
                    if profile_result.endswith('```'):
                        profile_result = profile_result[:-3]
                    profile_result = profile_result.strip()
                    
                    # Parse the cleaned JSON
                    profile_result = json.loads(profile_result)
                    
                    # Validate the structure
                    required_fields = ["basic_info", "professional_summary", "skills", "experience", "education"]
                    for field in required_fields:
                        if field not in profile_result:
                            raise ValueError(f"Missing required field: {field}")
                            
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse profile result as JSON: {str(e)}")
                logger.error(f"❌ Raw profile result: {profile_result}")
                return {
                    "status": "error",
                    "error": "Failed to parse profile result as JSON"
                }
            except ValueError as e:
                logger.error(f"❌ Invalid profile structure: {str(e)}")
                logger.error(f"❌ Profile result: {profile_result}")
                return {
                    "status": "error",
                    "error": f"Invalid profile structure: {str(e)}"
                }
            
            # Update candidate profile in Supabase
            profile_data = {
                "email": candidate_email,
                "profile": profile_result,
                "processing_status": {
                    "resume_processed": True,
                    "call_completed": False,
                    "last_updated": datetime.utcnow().isoformat()
                },
                **(additional_info or {})
            }
            
            # Store in Supabase
            await self.supabase.table('candidates_dev').update({
                'profile_json': profile_data,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', candidate_id).execute()
            
            logger.info("✅ Profile stored in Supabase")
            
            return {
                "status": "success",
                "candidate_id": candidate_id,
                "profile": profile_result
            }
            
        except Exception as e:
            logger.error(f"❌ Error in candidate intake process: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
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