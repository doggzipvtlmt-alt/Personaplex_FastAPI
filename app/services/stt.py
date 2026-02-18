from pathlib import Path

import httpx

from app.config import Settings


class STTService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def transcribe(self, audio_path: Path) -> str:
        if not self.settings.elevenlabs_api_key:
            return f"STT not configured. Received file: {audio_path.name}"

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": self.settings.elevenlabs_api_key}
        content_type = _detect_content_type(audio_path.suffix.lower())

        files = {
            "file": (audio_path.name, audio_path.read_bytes(), content_type),
            "model_id": (None, "scribe_v1"),
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(url, headers=headers, files=files)
            response.raise_for_status()
            payload = response.json()
        return (payload.get("text") or "").strip() or "No transcript returned from STT provider."


def _detect_content_type(suffix: str) -> str:
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".webm": "audio/webm",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")
