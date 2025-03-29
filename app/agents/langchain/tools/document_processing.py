from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import io
import os
import logging
from pydantic import Field
from .base import parse_llm_json_response

logger = logging.getLogger(__name__)

class PDFProcessor(BaseTool):
    """Tool for processing PDF documents."""
    
    name = "pdf_processor"
    description = "Processes PDF documents to extract text content"
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

    def _run(self, file_path: str) -> Dict[str, Any]:
        """Process a PDF file and extract its text content."""
        try:
            # Read PDF file
            reader = PdfReader(file_path)
            
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

    async def _arun(self, file_path: str) -> Dict[str, Any]:
        """Async version of PDF processing."""
        return self._run(file_path)

class ResumeParser(BaseTool):
    """Tool for parsing resume content."""
    
    name = "resume_parser"
    description = "Parses resume content to extract structured information"
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