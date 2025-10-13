from abc import ABC, abstractmethod
from llm.base import BaseLLM

class BaseAgent(ABC):
    """
    Base class for all LLM agents.
    """
    def __init__(self):
        self.agent_name: str = None
        self.description: str = None
        self.model: BaseLLM = None

    @abstractmethod
    def set_model(self, **kwargs) -> None:
        pass

    @abstractmethod
    def setup_agent(self, **kwargs) -> None:
        pass

    @abstractmethod
    def act(self, **kwargs) -> any:
        pass

    # @abstractmethod
    # def evaluate(self, **kwargs) -> any:
    #     pass