from typing import Any, Dict, List, Optional
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

class BaseChain:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ):
        """Initialize the base chain with a language model."""
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize tools and chains
        self.tools: List[BaseTool] = []
        self.chains: Dict[str, RunnableSequence] = {}
        
        # Initialize the chain
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the chain with tools and sub-chains."""
        raise NotImplementedError("Subclasses must implement _initialize_chain")
    
    def _initialize_tools(self):
        """Initialize the tools used by the chain."""
        raise NotImplementedError("Subclasses must implement _initialize_tools")
    
    def _initialize_chains(self):
        """Initialize the sub-chains used by the chain."""
        raise NotImplementedError("Subclasses must implement _initialize_chains")
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """Run the chain with the given inputs."""
        raise NotImplementedError("Subclasses must implement run")
    
    def _create_prompt(self, template: str, input_variables: List[str]) -> PromptTemplate:
        """Create a prompt template with the given template and input variables."""
        return PromptTemplate(
            input_variables=input_variables,
            template=template
        )
    
    def _create_chain(self, prompt: PromptTemplate) -> RunnableSequence:
        """Create a chain with the given prompt."""
        return prompt | self.llm 