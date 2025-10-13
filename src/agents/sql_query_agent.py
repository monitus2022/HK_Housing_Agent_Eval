from .base import BaseAgent
from llm import openrouter_llm_info, LLMExtraConfig
from prompts import LLMPromptTemplate
from logger import housing_logger
from utils import timer
from config import settings

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler


class SqlQueryAgent(BaseAgent):    
    """
    Agent that interacts with duckdb/sqlite SQL using simple prompts.
    """
    def __init__(self):
        super().__init__()
        self.agent_name: str = "Simple SQL Agent"
        self.description: str = "Agent that interacts with duckdb/sqlite SQL using simple prompts."

        self.model_name: Optional[str] = None
        self.model_id: Optional[str] = None
        self.db = None
        self.chain = None
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

    def setup_db(self, db_path: str, db_type: str="duckdb") -> None:
        """
        Setup database connection for SINGLE duckdb/sqlite.
        """
        if db_type == "duckdb":
            import duckdb
            self.db = duckdb.connect(database=db_path, read_only=True)
        elif db_type == "sqlite":
            import sqlite3
            self.db = sqlite3.connect(database=db_path, uri=False)
        else:
            housing_logger.error("Unsupported database type. Use 'duckdb' or 'sqlite'.")
            raise ValueError("Unsupported database type. Use 'duckdb' or 'sqlite'.")

    def _close_db(self) -> None:
        if self.db:
            self.db.close()
            self.db = None

    def setup_agent(self, model_params: Optional[LLMExtraConfig]) -> None:
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

        self.model: ChatOpenAI = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_api_url,
            model_name=self.model_id,
            callbacks=[self.token_count],
            **model_params
        )

        if not self.model:
            housing_logger.error("Failed to initialize the model.")
            raise ValueError("Failed to initialize the model.")
    
    @timer
    def act(self, prompt: LLMPromptTemplate) -> any:
        full_prompt = prompt.to_list()
        response = self.model.invoke(input=full_prompt)
        if not response:
            housing_logger.error("No response from the model.")
            raise ValueError("No response from the model.")
        return response