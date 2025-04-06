from typing import Any, Dict, List, Optional, Callable
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
import json
import logging # Add logging import

# Initialize logger for this module
logger = logging.getLogger(__name__)

def create_prompt(template: str, input_variables: List[str]) -> PromptTemplate:
    """Create a prompt template."""
    return PromptTemplate(
        template=template,
        input_variables=input_variables
    )

def create_chain(llm: ChatOpenAI, prompt: PromptTemplate) -> RunnableSequence:
    """Create a runnable sequence with the given prompt and LLM."""
    return prompt | llm

# Common utility function for parsing LLM responses
def parse_llm_json_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response into structured data."""
    try:
        # Find the start and end of the JSON object/array
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1:
             # Maybe it's an array?
             start = response.find("[")
             end = response.rfind("]")
             if start == -1 or end == -1:
                 raise ValueError("No JSON object or array found in the response")
        
        json_str = response[start:end+1]
        # Use json.loads() instead of eval() for safety and correctness
        parsed_json = json.loads(json_str)
        if not isinstance(parsed_json, dict):
             # Ensure we always return a dictionary, even if LLM returns a list etc.
             # Or handle this case differently if needed.
             logger.warning(f"LLM response was valid JSON but not a dictionary: {type(parsed_json)}")
             return {"parsed_content": parsed_json} # Wrap it
        return parsed_json
    except json.JSONDecodeError as json_err:
        logger.error(f"JSONDecodeError parsing LLM response: {json_err}")
        logger.error(f"LLM response snippet: {response[:500]}...") # Log snippet on error
        return {
            "error": f"Failed to parse LLM response (JSONDecodeError): {str(json_err)}"
        }
    except Exception as e:
        logger.error(f"Generic error parsing LLM response: {e}")
        logger.error(f"LLM response snippet: {response[:500]}...") # Log snippet on error
        return {
            "error": f"Failed to parse LLM response (Generic Error): {str(e)}"
        } 