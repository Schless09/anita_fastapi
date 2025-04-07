from typing import Any, Dict, List, Optional, Union, Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import io
import os
import logging
from pydantic import Field, BaseModel, PrivateAttr
from .base import parse_llm_json_response
from app.config.settings import Settings
import traceback
import asyncio

logger = logging.getLogger(__name__)

class PDFProcessorInput(BaseModel):
    """Input schema for the PDF processor tool."""
    file_path: str = Field(..., description="Path to the PDF file")
    extract_images: bool = Field(False, description="Whether to extract images from the PDF")

class ResumeParserInput(BaseModel):
    """Input schema for the resume parser tool."""
    file_path: str = Field(..., description="Path to the resume file")
    format: Optional[str] = Field(None, description="Format of the resume (pdf, docx, etc.)")

class PDFProcessor(BaseTool):
    """Tool for processing PDF documents."""
    
    name: str = "pdf_processor"
    description: str = """Useful for processing PDF documents and extracting text content.
    Input should be a JSON string with the following fields:
    - file_path: Path to the PDF file
    - extract_images: (optional) Whether to extract images from the PDF
    """
    args_schema: Type[BaseModel] = PDFProcessorInput
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

    def _quick_extract(self, file_input: Union[str, bytes]) -> Dict[str, Any]:
        """Quickly extract essential information from the first few pages of a PDF.
        
        Args:
            file_input: Either a file path (str) or PDF binary content (bytes)
        """
        try:
            logger.info("Starting quick PDF extraction")
            
            # Check if input is a file path or binary content
            if isinstance(file_input, str):
                # It's a file path
                logger.info(f"Quick extracting from PDF file path: {file_input}")
                try:
                    reader = PdfReader(file_input)
                    logger.info(f"Successfully opened PDF file: {file_input}")
                except FileNotFoundError as e:
                    logger.error(f"File not found: {file_input}")
                    return {
                        "status": "error",
                        "error": f"File not found: {str(e)}"
                    }
            else:
                # It's binary content
                logger.info("Quick extracting from PDF binary content")
                try:
                    pdf_stream = io.BytesIO(file_input)
                    reader = PdfReader(pdf_stream)
                    logger.info("Successfully opened PDF from binary content")
                except Exception as e:
                    logger.error(f"Error reading PDF from binary content: {str(e)}")
                    logger.error(f"PDF content length: {len(file_input)} bytes")
                    return {
                        "status": "error",
                        "error": f"Invalid PDF content: {str(e)}"
                    }
            
            # Extract text from first 3 pages only
            text_content = []
            total_pages = len(reader.pages)
            logger.info(f"Total pages in PDF: {total_pages}")
            
            for i, page in enumerate(reader.pages):
                if i >= 3:  # Only process first 3 pages
                    logger.info(f"Reached page limit (3), stopping extraction")
                    break
                try:
                    page_text = page.extract_text()
                    text_content.append(page_text)
                    logger.info(f"Successfully extracted text from page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                    continue
            
            # Combine all text
            quick_text = "\n".join(text_content)
            logger.info(f"Total extracted text length: {len(quick_text)} characters")
            
            # Log the extracted text for debugging
            logger.info(f"Extracted text for LLM: {quick_text[:200]}...")  # Log first 200 characters
            
            if not quick_text.strip():
                logger.error("No text content was extracted from the PDF")
                return {
                    "status": "error",
                    "error": "No text content was extracted from the PDF"
                }
            
            # Use LLM to extract essential info
            logger.info("Sending text to LLM for information extraction")
            prompt = f"""Extract the following essential information from this resume text (first few pages only):
            1. Current or most recent job title
            2. Current or most recent company
            
            Resume text:
            {quick_text}
            
            Return the information in a structured JSON format with these fields:
            - current_role: Current or most recent job title
            - current_company: Current or most recent company
            
            If any information is not found, use empty strings or empty arrays."""
            
            try:
                response = self.llm.invoke(prompt)
                logger.info("Successfully received response from LLM")
                
                # Log the LLM response for debugging
                logger.info(f"LLM response: {response.content}")
                
                # Parse the response into structured data
                essential_info = parse_llm_json_response(response.content)
                logger.info("Successfully parsed LLM response into structured data")
                
                return {
                    "status": "success",
                    "essential_info": essential_info,
                    "num_pages_processed": min(3, len(reader.pages))
                }
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}")
                logger.error(f"LLM response content: {response.content if 'response' in locals() else 'No response'}")
                return {
                    "status": "error",
                    "error": f"LLM processing failed: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Error in quick PDF extraction: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def process_pdf(self, file_input: Union[str, bytes]) -> Dict[str, Any]:
        """Process the entire PDF and extract all text content.
        
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
            
            # Extract text from all pages
            text_content = []
            for page in reader.pages:
                text_content.append(page.extract_text())
            
            # Combine all text
            full_text = "\n".join(text_content)
            
            return {
                "status": "success",
                "text": full_text,
                "num_pages": len(reader.pages)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

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
            
            # Log the raw LLM response before parsing
            logger.info(f"Raw LLM response for resume parsing: {response.content}")
            
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

    async def parse_resume(self, text_content: str) -> Dict[str, Any]:
        """Parse resume text and extract structured information.
        
        Args:
            text_content: The text content of the resume
        """
        try:
            # Optimize text content by removing excessive whitespace and newlines
            optimized_text = " ".join(text_content.split())
            logger.info(f"Optimized text length: {len(optimized_text)} characters")
            
            # Use LLM to extract structured information based on the desired profile_json structure
            prompt = f"""Analyze the following resume text and extract the specified information.
            
            Resume text:
            {optimized_text}
            
            Return ONLY a valid JSON object with the following structure and fields. 
            Use the specified types (string, array of strings, array of objects). 
            If information for a field is not found, use an appropriate empty value (empty string "", empty array [], or 0 for years_of_experience).
            
            {{
              "skills": ["string"],  // List of technical skills
              "education": [
                {{
                  "year": "string", // Year of graduation (if found, else "")
                  "degree": "string",
                  "institution": "string"
                }}
              ],
              "experience": [
                {{
                  "title": "string",
                  "company": "string",
                  "duration": "string", // Dates worked (e.g., "Nov 2021 – Present")
                  "description": "string" // Key responsibilities/achievements
                }}
              ],
              "current_role": "string", // Title of the most recent job
              "current_company": "string", // Company of the most recent job
              "years_of_experience": 0, // Attempt to infer total years, default to 0
              "professional_summary": "string", // A brief summary if available
              "additional_qualifications": [] // List of any other relevant qualifications (e.g., certifications, languages)
            }}
            
            Focus on accurately extracting data into the defined fields. Ensure the output is strictly a JSON object.
            """
            
            try:
                # Increase timeout to 120 seconds and add retries
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        response = await asyncio.wait_for(
                            self.llm.ainvoke(prompt),
                            timeout=120.0  # Increased timeout to 120 seconds
                        )
                        logger.info("Successfully received response from LLM")
                        break
                    except asyncio.TimeoutError as e:
                        retry_count += 1
                        last_error = e
                        logger.warning(f"LLM call timed out (attempt {retry_count}/{max_retries})")
                        if retry_count < max_retries:
                            await asyncio.sleep(5)  # Increased wait time between retries to 5 seconds
                        continue
                
                if retry_count == max_retries:
                    logger.error("LLM call timed out after all retries")
                    return {
                        "status": "error",
                        "error": "LLM processing timed out after multiple retries"
                    }
                
                # Parse the response into structured data
                parsed_data = parse_llm_json_response(response.content)
                logger.info("Successfully parsed LLM response into structured data")
                
                return {
                    "status": "success",
                    "profile": parsed_data
                }
                
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            } 