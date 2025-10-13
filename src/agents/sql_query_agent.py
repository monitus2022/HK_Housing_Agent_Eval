from .base import BaseAgent
from llm import openrouter_llm_info, LLMExtraConfig
from prompts import LLMPromptTemplate
from logger import housing_logger
from utils import timer
from config import settings
from db import DuckDBManager, QueryExecutor

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.chains.base import Chain
from langchain_core.messages import AIMessage
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import StrOutputParser


class SqlQueryAgent(BaseAgent):
    """
    Agent that interacts with duckdb SQL using simple prompts.
    """

    def __init__(self):
        super().__init__()
        self.agent_name: str = "Simple SQL Agent"
        self.description: str = (
            "Agent that interacts with duckdb SQL using simple prompts."
        )

        self.model_name: Optional[str] = None
        self.model_id: Optional[str] = None
        self.db = DuckDBManager()
        self.query_executor = QueryExecutor(self.db.conn)
        self.chain = None
        self.token_count = OpenAICallbackHandler()

    def set_model(
        self,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        if model_id:
            model_info = openrouter_llm_info.get_model_info_by_id(model_id)
            if not model_info:
                housing_logger.error(f"Model ID '{model_id}' not found.")
                raise ValueError(f"Model ID '{model_id}' not found.")
            self.model_name = model_info.name
        elif model_name:
            model_info = openrouter_llm_info.get_model_info_by_name(model_name)
            if not model_info:
                housing_logger.error(f"Model name '{model_name}' not found.")
                raise ValueError(f"Model name '{model_name}' not found.")
            self.model_id = model_info.id
            self.model_name = model_name
        else:
            housing_logger.error("Either model_name or model_id must be provided.")
            raise ValueError("Either model_name or model_id must be provided.")

    def setup_agent(self, model_params: Optional[LLMExtraConfig]) -> None:
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
            **model_params,
        )

        if not self.model:
            housing_logger.error("Failed to initialize the model.")
            raise ValueError("Failed to initialize the model.")

    @timer
    def act(self, prompt: LLMPromptTemplate) -> Optional[AIMessage]:
        full_prompt = prompt.to_list()
        response: AIMessage = self.model.invoke(input=full_prompt)
        if not response:
            housing_logger.error("No response from the model.")
            raise ValueError("No response from the model.")
        return response
   