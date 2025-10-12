from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class EvalSettings(BaseSettings):
    OPENROUTER_API_URL: str = Field(default="https://openrouter.ai/api/v1", env="OPENROUTER_API_URL")
    OPENROUTER_API_KEY: str = Field(default="", env="OPENROUTER_API_KEY")
    
    model_config = SettingsConfigDict(case_sensitive=False)

settings = EvalSettings()
