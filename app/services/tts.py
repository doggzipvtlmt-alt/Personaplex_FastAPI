import math
import struct
import wave
from pathlib import Path

import httpx

from app.config import Settings


class TTSService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def synthesize(self, text: str, output_path: Path) -> Path:
        if not self.settings.elevenlabs_api_key:
            return _generate_fallback_beep(output_path)

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}"
        headers = {
            "xi-api-key": self.settings.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.7},
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            audio_data = response.content

        output_path.write_bytes(audio_data)
        return output_path


def _generate_fallback_beep(output_path: Path) -> Path:
    sample_rate = 16000
    duration_seconds = 1.0
    frequency_hz = 440.0
    amplitude = 12000
    n_samples = int(sample_rate * duration_seconds)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(n_samples):
            sample = int(amplitude * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wav_file.writeframes(bytes(frames))

    return output_path
