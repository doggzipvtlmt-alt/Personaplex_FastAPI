from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    kb_url: str = Field(default="", alias="KB_URL")
    personaplex_url: str = Field(default="", alias="PERSONAPLEX_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    use_llm: bool = Field(default=False, alias="USE_LLM")
    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")
    output_dir: str = Field(default="./outputs", alias="OUTPUT_DIR")
    max_upload_bytes: int = 20 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
