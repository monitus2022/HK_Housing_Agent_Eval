from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class EvalSettings(BaseSettings):

    openrouter_api_url: str = Field(
        default="https://openrouter.ai/api/v1", env="OPENROUTER_API_URL"
    )
    openrouter_api_key: str = Field(env="OPENROUTER_API_KEY")

    sqlite_db_path: Optional[str] = Field(env="SQLITE_DB_PATH")
    duckdb_path: Optional[str] = Field(env="DUCKDB_PATH")

    llm_info_json_path: Optional[str] = Field(
        default="src/llm/model_info.json", env="LLM_INFO_JSON_PATH"
    )

    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=".env", env_file_encoding="utf-8"
    )


settings = EvalSettings()
