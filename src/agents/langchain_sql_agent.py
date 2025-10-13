from .base import BaseAgent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from config import settings
from typing import Optional
from llm import openrouter_llm_info, LLMExtraConfig
from prompts import LLMPromptTemplate
from logger import housing_logger
from utils import timer

class LangChainSqlAgent(BaseAgent):    
    """
    Agent that interacts with duckdb/sqlite SQL using LangChain's SQL Agent.
    """
    def __init__(self):
        super().__init__()
        self.agent_name: str = "LangChain SQL Agent"
        self.description: str = "Agent that interacts with duckdb/sqlite SQL using LangChain's SQL Agent."

        self.model_name: Optional[str] = None
        self.model_id: Optional[str] = None
        self.model: ChatOpenAI = None
        self.db: SQLDatabase = None
        self.toolkit: SQLDatabaseToolkit = None
        self.agent = None
        self.token_count = OpenAICallbackHandler()

    
    def set_model(self,
                  model_name: Optional[str] = None,
                  model_id: Optional[str] = None,
                  ) -> None:
        if model_id:
            # Verify if model_id allowed
            model_info = openrouter_llm_info.get_model_info_by_id(model_id)
            if not model_info:
                housing_logger.error(f"Model ID '{model_id}' not found.")
                raise ValueError(f"Model ID '{model_id}' not found.")
            self.model_name = model_info.name
        elif model_name:
            # Verify if model_name allowed
            model_info = openrouter_llm_info.get_model_info_by_name(model_name)
            if not model_info:
                housing_logger.error(f"Model name '{model_name}' not found.")
                raise ValueError(f"Model name '{model_name}' not found.")
            self.model_id = model_info.id
            self.model_name = model_name
        else:
            housing_logger.error("Either model_name or model_id must be provided.")
            raise ValueError("Either model_name or model_id must be provided.")

    def setup_agent(self, db_path: str, 
                    model_params: Optional[LLMExtraConfig]) -> None:
        """
        Setup SQL Agent for SINGLE duckdb/sqlite.
        """
        if not self.model_id:
            housing_logger.error("Model not set. Call set_model() first.")
            raise ValueError("Model not set. Call set_model() first.")
        
        if model_params:
            model_params = model_params.to_dict()
        else:
            model_params = {}

        self.model = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_api_url,
            model_name=self.model_id,
            **model_params,
        )
        self.db = SQLDatabase.from_uri(db_path)
        if not self.db:
            housing_logger.error("Failed to connect to the database.")
            raise ValueError("Failed to connect to the database.")

        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)

        self.agent = create_sql_agent(
            llm=self.model,
            toolkit=self.toolkit,
            verbose=True,  # Add verbose for debugging
            handle_parsing_errors=True,
            max_iterations=5
        )
        if not self.agent:
            housing_logger.error("Failed to create SQL agent.")
            raise ValueError("Failed to create SQL agent.")

    @timer
    def act(self, prompt: LLMPromptTemplate) -> any:
        if not self.agent:
            housing_logger.error("Agent not set up. Call setup_agent() first.")
            raise ValueError("Agent not set up. Call setup_agent() first.")
        
        response = self.agent.invoke(
            input=prompt.user_messages, 
            callbacks=[self.token_count]
            )

        if not response:
            housing_logger.error("Agent failed to produce a response.")
            raise ValueError("Agent failed to produce a response.")
        
        # Log token usage
        housing_logger.info(f"Prompt Tokens: {self.token_count.prompt_tokens}, "
                            f"Completion Tokens: {self.token_count.completion_tokens}, "
                            f"Total Tokens: {self.token_count.total_tokens}")
        
        return response