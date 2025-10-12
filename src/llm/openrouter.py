import json
from pydantic import BaseModel, Field
from .base import (
    BaseLLM,
    LLMInfo,
    LLMPromptConfig,
    LLMPromptTemplate,
)
from config import settings
from logger import housing_logger
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional
from utils import timer

MODEL_INFO_FILE_PATH = settings.llm_info_json_path


class OpenRouterLLMInfo(BaseModel):
    """
    Load Json info for OpenRouter models.
    """

    models: list[LLMInfo] = Field(default_factory=list)
    free_models: list[LLMInfo] = Field(default_factory=list)
    paid_models: list[LLMInfo] = Field(default_factory=list)

    # Load from JSON file
    @classmethod
    def load_from_json(cls) -> "OpenRouterLLMInfo":
        try:
            with open(MODEL_INFO_FILE_PATH, "r") as f:
                data = json.load(f)
        except Exception as e:
            housing_logger.error(f"Error loading JSON: {e}")
            return cls()
        models = [LLMInfo(**model) for model in data.get("openrouter", [])]
        free_models = [model for model in models if model.is_free]
        paid_models = [model for model in models if not model.is_free]
        housing_logger.info(f"{len(models)} models included in OpenRouter LLM info.")
        return cls(models=models, free_models=free_models, paid_models=paid_models)

    def get_model_info_by_name(self, model_name: str) -> Optional[LLMInfo]:
        for model in self.models:
            if model.name == model_name:
                return model
        return None

    def get_model_info_by_id(self, model_id: str) -> Optional[LLMInfo]:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def get_model_id_by_name(self, model_name: str) -> Optional[str]:
        model = self.get_model_info_by_name(model_name)
        return model.id if model else None

    def get_all_models(self) -> list[LLMInfo]:
        return self.models


openrouter_llm_info = OpenRouterLLMInfo.load_from_json()


class OpenRouterLLM(BaseLLM):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenRouter LLM with either model name or ID.
        """
        super().__init__()
        self.model_name = model_name
        self.model_id = model_id
        if not self.model_name and not self.model_id:
            housing_logger.error("Either model_name or model_id must be provided.")
            raise ValueError("Either model_name or model_id must be provided.")
        # Get id by name if only name provided
        if not model_id:
            self.model_id = openrouter_llm_info.get_model_id_by_name(model_name)
            # Verify model name available
            if not self.model_id:
                housing_logger.error(f"Model name '{model_name}' not found.")
                raise ValueError(f"Model name '{model_name}' not found.")

        self.api_key = settings.openrouter_api_key
        self.api_url = settings.openrouter_api_url
        if not self.api_key:
            housing_logger.error("OpenRouter API key is not set.")
            raise ValueError("OpenRouter API key is required.")

        self.kwargs = kwargs if kwargs else {}
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
        )

    def get_model_info(self) -> Optional[LLMInfo]:
        if self.model_name:
            return openrouter_llm_info.get_model_info_by_name(self.model_name)
        elif self.model_id:
            return openrouter_llm_info.get_model_info_by_id(self.model_id)
        return None

    @timer
    def prompt_model(self, prompt: LLMPromptTemplate) -> Optional[ChatCompletion]:
        prompt_messages = prompt.to_list()

        housing_logger.info(f"Sending prompt to OpenRouter model {self.model_id}.")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=prompt_messages,
            **self.kwargs,
        )
        if not response or not response.choices:
            housing_logger.error(
                f"No response from OpenRouter model {self.model_id}."
            )
            return None
        housing_logger.info(
            f"Received response from OpenRouter model {self.model_id}."
        )
        return response

    def parse_response(self, response: ChatCompletion) -> str:
        try:
            return response.choices[0].message.content
        except (KeyError, IndexError) as e:
            housing_logger.error(f"Error parsing response: {e}")
            return ""

    def test_model(self) -> bool:
        return False
