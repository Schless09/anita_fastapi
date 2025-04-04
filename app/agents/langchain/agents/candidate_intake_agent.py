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
from app.config.supabase import get_supabase_client
from app.config.utils import get_table_name
from app.config.settings import Settings
from postgrest import AsyncPostgrestClient
import json
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
import asyncio

logger = logging.getLogger(__name__)

# Define CandidateState model
class CandidateState(BaseModel):
    profile_json: Any # Or a more specific type if available
    status: str # Or an Enum if status values are fixed
    # Add other fields if needed by _save_state

# Define CandidateProfile model (based on expected structure)
class BasicInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None

class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None

class EducationItem(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None

class CandidateProfile(BaseModel):
    basic_info: Optional[BasicInfo] = None
    current_role: Optional[str] = None
    current_company: Optional[str] = None
    professional_summary: Optional[str] = None
    skills: Optional[List[str]] = None
    experience: Optional[List[ExperienceItem]] = None
    education: Optional[List[EducationItem]] = None
    additional_qualifications: Optional[List[str]] = None
    years_of_experience: Optional[float] = None # Use float to allow for numbers

class CandidateIntakeAgent(BaseAgent):
    def __init__(
        self,
        vector_service: VectorService,
        settings: Settings,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory: Optional[Any] = None,
        candidate_id: str = "",
        supabase: AsyncPostgrestClient = None
    ):
        super().__init__(model_name, temperature, memory)
        
        # Initialize tools as instance attributes
        self.pdf_processor = PDFProcessor()
        self.resume_parser = ResumeParser()
        self.vector_store = VectorStoreTool(vector_service=vector_service, settings=settings)
        self.email_tool = EmailTool()
        self.matching_tool = MatchingTool(vector_store=self.vector_store)
        
        # Initialize Supabase client
        self.supabase: AsyncPostgrestClient = supabase or get_supabase_client()
        
        # Store services for background processing
        self.vector_service = vector_service
        self.settings = settings
        
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
        
        self.candidate_id = candidate_id
        self.candidates_table = get_table_name("candidates")

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
            Pay close attention to extracting the full name, the title of the MOST RECENT job, and the company of the MOST RECENT job.
            
            You MUST return a valid JSON object with the following structure:
            {{
                \"basic_info\": {{
                    \"full_name\": \"string (Candidate's full name)\",
                    \"email\": \"string\",
                    \"phone\": \"string\",
                    \"location\": \"string\" 
                }},
                \"current_role\": \"string (Title of the most recent job)\", 
                \"current_company\": \"string (Company of the most recent job)\",
                \"professional_summary\": \"string\",
                \"skills\": [\"string\"],
                \"experience\": [
                    {{
                        \"title\": \"string\",
                        \"company\": \"string\",
                        \"duration\": \"string\",
                        \"description\": \"string\"
                    }}
                    # ... include all relevant experiences
                ],
                \"education\": [
                    {{
                        \"degree\": \"string\",
                        \"institution\": \"string\",
                        \"year\": \"string\"
                    }}
                ],
                \"additional_qualifications\": [\"string\"],
                \"years_of_experience\": number (Attempt to infer total years if possible, otherwise 0)
            }}
            
            Use the following data to create the profile:
            
            Parsed Data (May contain structured info):
            {parsed_data}
            
            Analysis (General assessment):
            {analysis}
            
            Prioritize information found in Parsed Data if available. Ensure the most recent job details are correctly identified for current_role and current_company.
            Return ONLY the JSON object, no other text or explanations."""
        )
        
        self.profile_creation_chain = profile_creation_prompt | self.llm

    async def process_candidate(
        self,
        resume_content: bytes,
        candidate_email: str,
        candidate_id: str
    ) -> Dict[str, Any]:
        """Process a new candidate submission."""
        try:
            # Step 1: Quick Resume Extraction
            logger.info("\nStep 1: ⚡ Quick Resume Extraction")
            logger.info("----------------------------------------")
            
            # First, check if candidate is already being processed
            candidate_status = await self.supabase.table(self.candidates_table)\
                .select('status')\
                .eq('id', candidate_id)\
                .single()\
                .execute()
            
            if candidate_status.data and candidate_status.data.get('status') in ['processing', 'completed']:
                logger.warning(f"Candidate {candidate_id} is already being processed or completed. Skipping duplicate processing.")
                return {
                    "status": "skipped",
                    "message": "Candidate is already being processed or completed"
                }

            # Extract first name from email for Retell
            full_name = candidate_email.split('@')[0].replace('.', ' ').title()
            first_name = full_name.split(' ')[0]

            # Process PDF with quick extraction
            quick_result = await self.pdf_processor._arun(
                "quick_extract",
                pdf_content=resume_content
            )

            if quick_result["status"] != "success":
                error_msg = quick_result.get("error", "Unknown error in quick extraction")
                logger.error(f"❌ Quick PDF Extraction Failed: {error_msg}")
                return quick_result
                
            # Store essential info immediately
            essential_info = quick_result["essential_info"]
            logger.info("Successfully extracted essential info:")
            logger.info(f"- Current Title: {essential_info.get('current_title', 'Not found')}")
            logger.info(f"- Current Company: {essential_info.get('current_company', 'Not found')}")
            logger.info(f"- Email: {essential_info.get('email', 'Not found')}")
            logger.info(f"- Phone: {essential_info.get('phone', 'Not found')}")
            logger.info(f"- Skills: {len(essential_info.get('skills', []))} skills found")
            
            # Add both full_name and first_name to essential_info
            essential_info["full_name"] = full_name
            essential_info["first_name"] = first_name
            logger.info("✅ Essential info extracted successfully")
            
            # Step 2: Trigger background processing
            logger.info("\nStep 2: 🚀 Triggering Background Processing")
            logger.info("----------------------------------------")
            
            # Start background processing
            logger.info("Creating background task for full resume processing")
            asyncio.create_task(self._process_full_resume_background(resume_content, candidate_id))
            
            logger.info("✅ Background processing triggered")
            
            # Return immediately with essential info
            return {
                "status": "success",
                "candidate_id": candidate_id,
                "essential_info": essential_info,
                "message": "Full resume processing started in background"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in candidate intake process: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _process_full_resume_background(
        self,
        resume_content: bytes,
        candidate_id: str
    ) -> None:
        """Process the full resume in the background."""
        try:
            logger.info(f"\n=== Starting Background Processing for Candidate {candidate_id} ===")
            
            # Update status to processing
            logger.info("Updating candidate status to 'processing'")
            await self.supabase.table(self.candidates_table).update({
                "status": "processing",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", candidate_id).execute()
            logger.info("Status updated successfully")

            # Process the full resume in chunks
            try:
                logger.info("Starting full PDF processing")
                # First get the text content
                result = await self.pdf_processor.process_pdf(resume_content)
                if result["status"] != "success":
                    error_msg = result.get("error", "Unknown error in PDF processing")
                    logger.error(f"Failed to process PDF: {error_msg}")
                    raise Exception(f"Failed to process PDF: {error_msg}")

                logger.info(f"Successfully processed PDF with {result.get('num_pages', 0)} pages")
                
                # Parse the resume in chunks to avoid timeout
                logger.info("Starting chunked text processing")
                text_chunks = self._chunk_text(result["text"])
                logger.info(f"Split text into {len(text_chunks)} chunks")
                
                parsed_data = {}
                max_retries = 3
                
                for i, chunk in enumerate(text_chunks):
                    logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                    retry_count = 0
                    chunk_success = False
                    
                    while retry_count < max_retries and not chunk_success:
                        try:
                            chunk_result = await self.resume_parser.parse_resume(chunk)
                            if chunk_result["status"] == "success":
                                logger.info(f"Successfully parsed chunk {i+1}")
                                # Merge the parsed data
                                parsed_data = self._merge_parsed_data(parsed_data, chunk_result["profile"])
                                chunk_success = True
                            else:
                                error_msg = chunk_result.get("error", "Unknown error")
                                logger.warning(f"Failed to parse chunk {i+1} (attempt {retry_count + 1}/{max_retries}): {error_msg}")
                                retry_count += 1
                                if retry_count < max_retries:
                                    await asyncio.sleep(1)  # Wait before retrying
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout processing chunk {i+1} (attempt {retry_count + 1}/{max_retries})")
                            retry_count += 1
                            if retry_count < max_retries:
                                await asyncio.sleep(1)  # Wait before retrying
                        except Exception as e:
                            logger.error(f"Error processing chunk {i+1}: {str(e)}")
                            retry_count += 1
                            if retry_count < max_retries:
                                await asyncio.sleep(1)  # Wait before retrying
                    
                    if not chunk_success:
                        logger.error(f"Failed to process chunk {i+1} after {max_retries} attempts")
                        # Continue with next chunk even if this one failed

                logger.info("Completed chunk processing")
                logger.info(f"Final parsed data contains {len(parsed_data)} fields")

                # Update status to completed
                logger.info("Updating candidate status to 'completed'")
                await self.supabase.table(self.candidates_table).update({
                    "status": "completed",
                    "profile_json": parsed_data,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", candidate_id).execute()
                logger.info("Status updated successfully")

            except asyncio.TimeoutError:
                logger.error("Background processing timed out")
                await self.supabase.table(self.candidates_table).update({
                    "status": "error",
                    "error_message": "Processing timed out",
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", candidate_id).execute()
            except Exception as e:
                logger.error(f"Error in background resume processing: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                await self.supabase.table(self.candidates_table).update({
                    "status": "error",
                    "error_message": str(e),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", candidate_id).execute()

        except Exception as e:
            logger.error(f"Error updating candidate status: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _chunk_text(self, text: str, chunk_size: int = 10000) -> List[str]:
        """Split text into manageable chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _merge_parsed_data(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge parsed data from different chunks."""
        merged = existing.copy() if existing else {}
        
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list):
                # For lists, append unique items
                if key not in merged:
                    merged[key] = []
                merged[key].extend([item for item in value if item not in merged[key]])
            elif isinstance(value, dict):
                # For dicts, recursively merge
                if key not in merged:
                    merged[key] = {}
                merged[key].update(value)
            else:
                # For other types, prefer non-empty values
                if not merged[key] and value:
                    merged[key] = value
        
        return merged

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

    async def _save_state(self, state: CandidateState):
        try:
            await self.supabase.table(self.candidates_table).update({
                'profile_json': json.loads(state.profile_json) if isinstance(state.profile_json, str) else state.profile_json,
                'status': state.status,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', self.candidate_id).execute()
        except Exception as e:
            logger.error(f"❌ Error saving state: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}") 