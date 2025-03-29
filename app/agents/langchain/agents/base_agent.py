from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        memory: Optional[ConversationBufferMemory] = None
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools: List[BaseTool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        self.emoji = "🤖"  # Default robot emoji
        logger.info(f"{self.emoji} Initialized {self.__class__.__name__}")

    def _create_prompt(self, system_message: str) -> ChatPromptTemplate:
        """Create a prompt template for the agent."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def _initialize_agent(self, system_message: str):
        """Initialize the agent with tools and prompt."""
        prompt = self._create_prompt(system_message)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            ),
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    async def run(self, input_text: str) -> Dict[str, Any]:
        """Run the agent with the given input."""
        if not self.agent_executor:
            raise ValueError("Agent not initialized. Call _initialize_agent first.")
        return await self.agent_executor.ainvoke({"input": input_text})

    def _log_start(self, action: str, **kwargs):
        """Log the start of an action with the agent's emoji."""
        logger.info(f"{self.emoji} Starting {action} with params: {kwargs}")

    def _log_success(self, action: str, result: Any):
        """Log successful completion of an action with the agent's emoji."""
        logger.info(f"{self.emoji} Successfully completed {action}")
        logger.debug(f"{self.emoji} Result: {result}")

    def _log_error(self, action: str, error: Exception):
        """Log an error with the agent's emoji."""
        logger.error(f"{self.emoji} Error in {action}: {str(error)}")
        logger.error(f"{self.emoji} Full traceback: {traceback.format_exc()}")

    async def _run(self, task: str) -> Dict[str, Any]:
        """Implementation of run in subclasses."""
        raise NotImplementedError("Subclasses must implement _run")

    def _store_interaction(
        self,
        job_id: Optional[str],
        candidate_id: str,
        interaction_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Store an interaction in the conversation history."""
        self._log_start("store_interaction", 
                       job_id=job_id, 
                       candidate_id=candidate_id, 
                       interaction_type=interaction_type)
        
        try:
            if job_id:
                if job_id not in self.conversation_history:
                    self.conversation_history[job_id] = {
                        "candidate_id": candidate_id,
                        "interactions": []
                    }
                
                self.conversation_history[job_id]["interactions"].append({
                    "type": interaction_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                })
            else:
                # Store by candidate_id if no job_id (e.g., for calls)
                if candidate_id not in self.conversation_history:
                    self.conversation_history[candidate_id] = {
                        "interactions": []
                    }
                
                self.conversation_history[candidate_id]["interactions"].append({
                    "type": interaction_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                })
            
            self._log_success("store_interaction", None)
        except Exception as e:
            self._log_error("store_interaction", e)
            raise 