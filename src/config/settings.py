from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from logger import housing_logger


class EvalSettings(BaseSettings):

    housing_logger.info(f"CWD: {os.getcwd()}")
    housing_logger.info(f".env exists: {os.path.exists('.env')}")

    openrouter_api_url: str = Field(
        default="https://openrouter.ai/api/v1", env="OPENROUTER_API_URL"
    )
    openrouter_api_key: str = Field(env="OPENROUTER_API_KEY")

    duckdb_path: str = Field(env="DUCKDB_PATH")

    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=".env", env_file_encoding="utf-8"
    )


settings = EvalSettings()
