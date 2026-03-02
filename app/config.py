from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="PersonaPlex Voice Assistant", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID")

    kb_base_url: str | None = Field(default=None, alias="KB_BASE_URL")
    kb_api_key: str | None = Field(default=None, alias="KB_API_KEY")

    personaplex_base_url: str | None = Field(default=None, alias="PERSONAPLEX_BASE_URL")

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str | None = Field(default=None, alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str | None = Field(default=None, alias="PINECONE_NAMESPACE")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")
    output_dir: str = Field(default="/tmp/personaplex_outputs", alias="OUTPUT_DIR")
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    allowed_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        alias="ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
