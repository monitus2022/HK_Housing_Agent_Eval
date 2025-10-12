from abc import ABC, abstractmethod
import json
from pydantic import BaseModel, Field

class LLMInfo(BaseModel):
    name: str = Field(..., description="Name of the model")
    parameters: str = Field("", description="Parameters of the model")
    id: str = Field(..., description="ID of the model")
    description: str = Field("", description="Description of the model")
    input_cost: float = Field(0.0, description="Input cost per 1M tokens")
    output_cost: float = Field(0.0, description="Output cost per 1M tokens")

class OpenRouterLLMInfo(BaseModel):
    free_models: list[LLMInfo] = Field(..., description="List of free models")
    paid_models: list[LLMInfo] = Field(..., description="List of paid models")
    
    # Load from JSON file
    @classmethod
    def load_from_json(cls, file_path: str) -> "OpenRouterLLMInfo":
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return cls(**data)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return cls()

class BaseLLM(ABC):
    def __init__(self):
        self.model_name: str = None
        self.api_url = None
        self.api_key = None
        self.kwargs = {}
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass
    
