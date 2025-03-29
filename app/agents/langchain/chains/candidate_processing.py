from typing import Any, Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from ..tools.document_processing import PDFProcessor, ResumeParser
from ..tools.vector_store import VectorStoreTool
from ..tools.communication import EmailTool
import logging

logger = logging.getLogger(__name__)

class CandidateProcessingChain:
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
        self.pdf_processor = PDFProcessor()
        self.resume_parser = ResumeParser()
        
        if vector_store:
            logger.info("CandidateProcessingChain using provided VectorStoreTool instance")
            self.vector_store = vector_store
        else:
            logger.warning("⚠️ CandidateProcessingChain creating new VectorStoreTool - this should be avoided!")
            self.vector_store = VectorStoreTool()
            
        self.email_tool = EmailTool()
        
        # Initialize chains
        self._initialize_chains()

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
            template="""Create a comprehensive candidate profile using the parsed data and analysis:
            
            Parsed Data:
            {parsed_data}
            
            Analysis:
            {analysis}
            
            Create a profile in JSON format with:
            1. Basic information
            2. Professional summary
            3. Skills and expertise
            4. Experience highlights
            5. Education details
            6. Additional qualifications"""
        )
        
        self.profile_creation_chain = profile_creation_prompt | self.llm

    async def process_candidate(
        self,
        resume_path: str,
        candidate_email: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a new candidate submission."""
        try:
            # Step 1: Process PDF
            pdf_result = await self.pdf_processor._arun(resume_path)
            if pdf_result["status"] != "success":
                return pdf_result
            
            # Step 2: Parse resume
            parse_result = await self.resume_parser._arun(pdf_result["text_content"])
            if parse_result["status"] != "success":
                return parse_result
            
            # Step 3: Analyze resume
            analysis_result = await self.resume_analysis_chain.ainvoke(
                {"resume_text": pdf_result["text_content"]}
            )
            
            # Step 4: Create profile
            profile_result = await self.profile_creation_chain.ainvoke(
                {"parsed_data": parse_result["parsed_data"], "analysis": analysis_result}
            )
            
            # Step 5: Store in vector store
            profile_data = {
                "email": candidate_email,
                "profile": profile_result,
                **(additional_info or {})
            }
            
            vector_result = await self.vector_store._arun(
                "store_candidate",
                candidate_data=profile_data
            )
            
            if vector_result["status"] != "success":
                return vector_result
            
            # Step 6: Send confirmation email
            email_result = await self.email_tool._arun(
                "send_email",
                to_email=candidate_email,
                subject="Resume Processing Complete",
                content=f"""Dear {profile_data.get('name', 'Candidate')},

Thank you for submitting your resume. We have successfully processed your application and created your candidate profile.

Next Steps:
1. Our team will review your profile
2. We will match you with relevant job opportunities
3. You will receive updates on potential matches

Best regards,
The Recruitment Team"""
            )
            
            return {
                "status": "success",
                "candidate_id": vector_result.get("id"),
                "profile": profile_result,
                "email_sent": email_result.get("success")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 