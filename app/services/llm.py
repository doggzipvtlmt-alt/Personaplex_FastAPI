from app.clients import OpenAIClient, PersonaPlexClient


class LLMService:
    def __init__(self, openai_client: OpenAIClient, personaplex_client: PersonaPlexClient):
        self.openai_client = openai_client
        self.personaplex_client = personaplex_client

    async def generate(self, transcript: str, context: str = "") -> str:
        if self.openai_client.client:
            answer = self.openai_client.generate_answer(transcript=transcript, context=context)
            if answer:
                return answer

        personaplex_answer = await self.personaplex_client.generate(transcript=transcript, context=context)
        if personaplex_answer:
            return personaplex_answer

        return f"LLM not configured. Transcript: {transcript}"
