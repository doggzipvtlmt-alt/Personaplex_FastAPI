from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class TextAgentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_prompt: str = Field(default="NATF2.pt")
    top_k: int = Field(default=5, ge=1, le=20)
    return_mode: str = Field(default="file", pattern="^(file|url)$")


class KBSearchResult(BaseModel):
    source: str | None = None
    text: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None


class ResponsePayload(BaseModel):
    answer: str
    citations: list[str]


class AgentUrlResponse(BaseModel):
    job_id: str
    audio_url: str
    transcript: str
    kb_sources: list[str]


class TranscriptPayload(BaseModel):
    text: str
    raw: dict[str, Any]


class JobJsonResponse(BaseModel):
    job_id: str
    transcript: str
    kb_results: list[dict[str, Any]]
    response: ResponsePayload
    timings_ms: dict[str, int]
