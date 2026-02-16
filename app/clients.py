import asyncio
import io
import json
import wave
from typing import Any

import httpx
from openai import OpenAI

from app.config import Settings


class HTTPClientBase:
    def __init__(self, settings: Settings):
        self.timeout = settings.request_timeout_seconds

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        retries: int = 1,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Request failed for {url}: {last_error}")


class PersonaPlexClient(HTTPClientBase):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.base_url = settings.personaplex_url.rstrip("/")

    async def stt(self, audio_bytes: bytes, filename: str = "input.wav") -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("PERSONAPLEX_URL is not configured")
        url = f"{self.base_url}/v1/stt"
        files = {"file": (filename, audio_bytes, "audio/wav")}
        response = await self._request_with_retry("POST", url, files=files, retries=1)
        return response.json()

    async def tts(self, text: str, voice_prompt: str) -> bytes:
        if not self.base_url:
            raise RuntimeError("PERSONAPLEX_URL is not configured")
        url = f"{self.base_url}/v1/tts"
        payload = {"text": text, "voice_prompt": voice_prompt}
        try:
            response = await self._request_with_retry("POST", url, json=payload, retries=1)
        except RuntimeError:
            return await self.s2s_fallback(text=text, voice_prompt=voice_prompt)

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body = response.json()
            audio_url = body.get("audio_url") or body.get("url") or body.get("download_url")
            if not audio_url:
                raise RuntimeError("TTS JSON response missing audio URL")
            audio_resp = await self._request_with_retry("GET", audio_url, retries=1)
            return audio_resp.content
        return response.content

    async def s2s_fallback(self, text: str, voice_prompt: str) -> bytes:
        if not self.base_url:
            raise RuntimeError("PERSONAPLEX_URL is not configured")
        url = f"{self.base_url}/v1/s2s"
        silent_wav = make_silent_wav(duration_ms=300)
        data = {"text": text, "voice_prompt": voice_prompt}
        files = {"file": ("silent.wav", silent_wav, "audio/wav")}
        response = await self._request_with_retry("POST", url, data=data, files=files, retries=1)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body = response.json()
            audio_url = body.get("audio_url") or body.get("url") or body.get("download_url")
            if not audio_url:
                raise RuntimeError("S2S fallback JSON response missing audio URL")
            audio_resp = await self._request_with_retry("GET", audio_url, retries=1)
            return audio_resp.content
        return response.content


def make_silent_wav(duration_ms: int = 300, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_samples)
    return buffer.getvalue()


class KBClient(HTTPClientBase):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.base_url = settings.kb_url.rstrip("/")

    async def search(self, query: str, top_k: int) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("KB_URL is not configured")
        url = f"{self.base_url}/kb/search"
        payload = {"query": query, "top_k": top_k}
        response = await self._request_with_retry("POST", url, json=payload, retries=1)
        return response.json()


class OpenAIClient:
    SYSTEM_PROMPT = (
        "You are HR of xccelera.ai. Answer ONLY from provided KB context. "
        "If missing, ask a question."
    )

    def __init__(self, settings: Settings):
        self.model = settings.openai_model
        self.enabled = settings.use_llm
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate_answer(self, query: str, kb_context: str) -> str:
        if not self.enabled:
            raise RuntimeError("LLM client is disabled")
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is required when USE_LLM=true")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "assistant", "content": f"KB Context:\n{kb_context}"},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content or ""
        return content.strip()


def normalize_kb_items(raw_kb: Any) -> list[dict[str, Any]]:
    if isinstance(raw_kb, list):
        return [item for item in raw_kb if isinstance(item, dict)]
    if isinstance(raw_kb, dict):
        if isinstance(raw_kb.get("results"), list):
            return [item for item in raw_kb["results"] if isinstance(item, dict)]
        if isinstance(raw_kb.get("data"), list):
            return [item for item in raw_kb["data"] if isinstance(item, dict)]
    try:
        parsed = json.loads(raw_kb)
        return normalize_kb_items(parsed)
    except Exception:
        return []
