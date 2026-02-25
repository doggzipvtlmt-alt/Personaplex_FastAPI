import asyncio
import json
from typing import Any

import httpx
from openai import OpenAI
from openai import OpenAIError

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
                await asyncio.sleep(0.4 * (attempt + 1))
        raise RuntimeError(f"Request failed for {url}: {last_error}")


class KBClient(HTTPClientBase):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.base_url = (settings.kb_base_url or "").rstrip("/")
        self.api_key = settings.kb_api_key

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.base_url:
            return []

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"query": query, "top_k": top_k}
        response = await self._request_with_retry(
            "POST",
            f"{self.base_url}/kb/search",
            json=payload,
            headers=headers,
            retries=1,
        )
        return normalize_kb_items(response.json())


class PersonaPlexClient(HTTPClientBase):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.base_url = (settings.personaplex_base_url or "").rstrip("/")

    async def generate(self, transcript: str, context: str = "") -> str:
        if not self.base_url:
            return ""

        payload = {"transcript": transcript, "context": context}
        response = await self._request_with_retry(
            "POST",
            f"{self.base_url}/v1/generate",
            json=payload,
            retries=1,
        )
        data = response.json()
        return (data.get("answer") or data.get("text") or "").strip()


class OpenAIClient:
    SYSTEM_PROMPT = "You are a concise and helpful assistant."

    def __init__(self, settings: Settings):
        api_key = (settings.openai_api_key or "").strip()
        if not api_key or api_key in {"your_key_here", "your_key*here"}:
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = settings.openai_model

    def generate_answer(self, transcript: str, context: str = "") -> str:
        if not self.client:
            return ""

        messages: list[dict[str, str]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "assistant", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": transcript})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
            )
            return (completion.choices[0].message.content or "").strip()
        except OpenAIError:
            return ""


def normalize_kb_items(raw_kb: Any) -> list[dict[str, Any]]:
    if isinstance(raw_kb, list):
        return [item for item in raw_kb if isinstance(item, dict)]
    if isinstance(raw_kb, dict):
        for key in ("results", "data", "items"):
            value = raw_kb.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(raw_kb, str):
        try:
            return normalize_kb_items(json.loads(raw_kb))
        except Exception:
            return []
    return []
