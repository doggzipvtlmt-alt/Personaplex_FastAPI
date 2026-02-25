import os
import httpx

class PersonaPlexHTTPClient:
    def __init__(self):
        self.base_url = os.getenv("https://deds0k3ekp5j3d-8000.proxy.runpod.net/")
        if not self.base_url:
            raise RuntimeError("Missing PERSONAPLEX_URL env var")

    async def tts(self, text: str, voice_prompt: str = "NATF2.pt"):
        url = f"{self.base_url}/tts"
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(url, json={"text": text, "voice_prompt": voice_prompt})
            r.raise_for_status()
            return r.json()
