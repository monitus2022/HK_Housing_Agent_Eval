from pydantic import BaseModel, Field
from typing import Optional

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

    def to_str(self) -> str:
        """Convert to single string for LLM input."""
        parts = []
        if self.system_messages:
            parts.append(f"System: {self.system_messages}")
        parts.append(f"User: {self.user_messages}")
        if self.assistant_messages:
            parts.append(f"Assistant: {self.assistant_messages}")
        return "\n\n".join(parts)