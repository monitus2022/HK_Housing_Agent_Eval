import json
from pydantic import BaseModel, Field
from .base import BaseLLM, LLMInfo
from config import settings
from logger import housing_logger
from openai import OpenAI


class OpenRouterLLMInfo(BaseModel):
    models: list[LLMInfo] = Field(default_factory=list)
    free_models: list[LLMInfo] = Field(default_factory=list)
    paid_models: list[LLMInfo] = Field(default_factory=list)

    model_info_file_path = settings.llm_info_json_path

    # Load from JSON file
    @classmethod
    def load_from_json(cls) -> "OpenRouterLLMInfo":
        try:
            with open(cls.model_info_file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            housing_logger.error(f"Error loading JSON: {e}")
            return cls()
        cls.models = [LLMInfo(**model) for model in data.get("models", [])]
        cls.free_models = [model for model in cls.models if model.is_free]
        cls.paid_models = [model for model in cls.models if not model.is_free]

    def get_model_by_name(self, model_name: str) -> LLMInfo | None:
        for model in self.models:
            if model.name == model_name:
                return model
        return None

    def list_all_models(self) -> dict[str, list[str]]:
        return self.models


openrouter_llm_info = OpenRouterLLMInfo.load_from_json()


class OpenRouterLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.api_key = settings.openrouter_api_key
        self.api_url = settings.openrouter_api_url
        self.kwargs = kwargs

        if not self.api_key:
            housing_logger.error("OpenRouter API key is not set.")
            raise ValueError("OpenRouter API key is required.")

        # Verify model name available
        if not openrouter_llm_info.get_model_by_name(self.model_name):
            housing_logger.error(f"Model '{self.model_name}' is not available.")
            raise ValueError(f"Model '{self.model_name}' is not available.")

        self.client = OpenAI(
            api_key=self.api_key,
            api_base=self.api_url,
        )

    def get_model_info(self) -> LLMInfo | None:
        return openrouter_llm_info.get_model_by_name(self.model_name)

    def prompt_model(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.kwargs,
        )
        if not response or "choices" not in response or len(response["choices"]) == 0:
            housing_logger.error("No response from OpenRouter model.")
            return ""
        return response

    def parse_response(self, response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            housing_logger.error(f"Error parsing response: {e}")
            return ""

    def test_model(self) -> bool:
        return False
