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


class LLMPromptTemplate(BaseModel):
    user_messages: str = Field(..., description="User messages")
    system_messages: Optional[str] = Field(None, description="System messages")
    assistant_messages: Optional[str] = Field(None, description="Assistant messages")

    def to_list(self) -> list:
        """Convert to list of messages for LLM input."""
        messages = [{"role": "user", "content": self.user_messages}]
        if self.system_messages:
            messages.insert(0, {"role": "system", "content": self.system_messages})
        if self.assistant_messages:
            messages.append({"role": "assistant", "content": self.assistant_messages})
        return messages


class LLMExtraConfig(BaseModel):
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


class BaseLLM(ABC):
    """
    Base class for all LLMs.
    """
    def __init__(self):
        self.model_name: str = None
        self.api_url: str = None
        self.api_key: str = None
        self.kwargs: Optional[dict] = None

    @abstractmethod
    def get_model_info(self) -> Optional[LLMInfo]:
        pass

    @abstractmethod
    def prompt_model(self, prompt: LLMPromptTemplate) -> str:
        pass

    @abstractmethod
    def test_model(self) -> bool:
        return False
