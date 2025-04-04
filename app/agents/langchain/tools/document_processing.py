from typing import Any, Dict, List, Optional, Union, Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import io
import os
import logging
from pydantic import Field, BaseModel, PrivateAttr
from .base import parse_llm_json_response

logger = logging.getLogger(__name__)

class PDFProcessorInput(BaseModel):
    file_path: str
    extract_images: bool = False

class ResumeParserInput(BaseModel):
    file_path: str
    format: str = None

class PDFProcessor(BaseTool):
    """Tool for processing PDF documents."""
    
    name: str = "pdf_processor"
    description: str = """Useful for processing PDF documents and extracting text content.
    Input should be a JSON string with the following fields:
    - file_path: Path to the PDF file
    - extract_images: (optional) Whether to extract images from the PDF
    """
    args_schema: Type[PDFProcessorInput] = PDFProcessorInput
    return_direct: bool = True
    _settings: Settings = PrivateAttr()
    # Define fields that will be set in __init__
    llm: ChatOpenAI = Field(default=None)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self):
        """Initialize the PDF processing tool."""
        super().__init__()
        
        # Set up LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.3
        )

    def _run(self, file_input: Union[str, bytes]) -> Dict[str, Any]:
        """Process a PDF file and extract its text content.
        
        Args:
            file_input: Either a file path (str) or PDF binary content (bytes)
        """
        try:
            # Check if input is a file path or binary content
            if isinstance(file_input, str):
                # It's a file path
                logger.info(f"Processing PDF from file path: {file_input}")
                try:
                    reader = PdfReader(file_input)
                except FileNotFoundError as e:
                    logger.error(f"File not found: {file_input}")
                    return {
                        "status": "error",
                        "error": f"File not found: {str(e)}"
                    }
            else:
                # It's binary content
                logger.info("Processing PDF from binary content")
                try:
                    pdf_stream = io.BytesIO(file_input)
                    reader = PdfReader(pdf_stream)
                except Exception as e:
                    logger.error(f"Error reading PDF from binary content: {str(e)}")
                    return {
                        "status": "error",
                        "error": f"Invalid PDF content: {str(e)}"
                    }
            
            # Extract text from each page
            text_content = []
            for page in reader.pages:
                text_content.append(page.extract_text())
            
            # Combine all text
            full_text = "\n".join(text_content)
            
            return {
                "status": "success",
                "text_content": full_text,
                "num_pages": len(reader.pages)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _arun(self, file_input: Union[str, bytes]) -> Dict[str, Any]:
        """Async version of PDF processing."""
        return self._run(file_input)

class ResumeParser(BaseTool):
    """Tool for parsing resumes and extracting structured information."""
    
    name: str = "resume_parser"
    description: str = """Useful for parsing resumes and extracting structured information.
    Input should be a JSON string with the following fields:
    - file_path: Path to the resume file
    - format: (optional) Format of the resume (pdf, docx, etc.)
    """
    args_schema: Type[ResumeParserInput] = ResumeParserInput
    return_direct: bool = True
    _settings: Settings = PrivateAttr()
    # Define fields that will be set in __init__
    llm: ChatOpenAI = Field(default=None)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self):
        """Initialize the resume parsing tool."""
        super().__init__()
        
        # Set up LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.3
        )

    def _run(self, text_content: str) -> Dict[str, Any]:
        """Parse resume text and extract structured information."""
        try:
            # Use LLM to extract structured information
            prompt = f"""Extract the following information from this resume text:
            1. Contact Information
            2. Professional Summary
            3. Work Experience
            4. Education
            5. Skills
            6. Certifications
            
            Resume text:
            {text_content}
            
            Return the information in a structured JSON format."""
            
            response = self.llm.invoke(prompt)
            
            # Parse the response into structured data
            parsed_data = parse_llm_json_response(response.content)
            
            return {
                "status": "success",
                "parsed_data": parsed_data
            }
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _arun(self, text_content: str) -> Dict[str, Any]:
        """Async version of resume parsing."""
        return self._run(text_content) 