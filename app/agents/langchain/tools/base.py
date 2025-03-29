from typing import Any, Dict, List, Optional, Callable
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

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
        # Extract JSON from response
        json_str = response[response.find("{"):response.rfind("}")+1]
        return eval(json_str)
    except Exception as e:
        return {
            "error": f"Failed to parse LLM response: {str(e)}"
        } 