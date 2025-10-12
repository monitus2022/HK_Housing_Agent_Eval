from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Optional


class LLMInfo(BaseModel):
    name: str = Field(..., description="Name of the model")
    parameters: str = Field("", description="Parameters of the model")
    id: str = Field(..., description="ID of the model")
    description: str = Field("", description="Description of the model")
    input_cost: float = Field(0.0, description="Input cost per 1M tokens")
    output_cost: float = Field(0.0, description="Output cost per 1M tokens")
    is_free: bool = Field(True, description="Is the model free to use")


class BaseLLM(ABC):
    def __init__(self):
        self.model_name: str = None
        self.api_url: str = None
        self.api_key: str = None
        self.kwargs: Optional[dict] = None

    @abstractmethod
    def prompt_model(self, prompt: str) -> str:
        pass

    @abstractmethod
    def test_model(self) -> bool:
        return False


class LLMPromptTemplate(BaseModel):
    system_prompt: str = Field(..., description="System prompt for the LLM")
    user_prompt: str = Field(..., description="Template for user prompt")
    assistant_prompt: str = Field(..., description="Template for assistant response")


class PromptConfig(BaseModel):
    # Optional depends on use case and LLM capabilities
    temperature: Optional[float] = Field(
        None, description="Controls randomness (0.0 to 1.0)"
    )
    top_p: Optional[float] = Field(None, description="Nucleus sampling (0.0 to 1.0)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    frequency_penalty: Optional[float] = Field(
        None, description="Reduces repetition (-2.0 to 2.0)"
    )
    presence_penalty: Optional[float] = Field(
        None, description="Encourages new topics (-2.0 to 2.0)"
    )

    def to_dict(self) -> dict:
        """Convert to dict for logging/serialization, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
