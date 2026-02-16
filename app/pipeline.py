import time
from dataclasses import dataclass
from typing import Any

from app.clients import KBClient, OpenAIClient, PersonaPlexClient, normalize_kb_items
from app.config import Settings
from app.storage import JobStorage


@dataclass
class PipelineResult:
    job_id: str
    transcript: str
    kb_results: list[dict[str, Any]]
    response_text: str
    citations: list[str]
    audio_bytes: bytes
    timings_ms: dict[str, int]


class VoicePipeline:
    def __init__(
        self,
        settings: Settings,
        storage: JobStorage,
        personaplex_client: PersonaPlexClient,
        kb_client: KBClient,
        openai_client: OpenAIClient,
    ):
        self.settings = settings
        self.storage = storage
        self.personaplex_client = personaplex_client
        self.kb_client = kb_client
        self.openai_client = openai_client

    async def run_from_audio(
        self,
        audio_bytes: bytes,
        voice_prompt: str,
        top_k: int,
        input_filename: str = "input.wav",
    ) -> PipelineResult:
        overall_start = time.perf_counter()
        job_id = self.storage.create_job()
        self.storage.save_bytes(job_id, "input.wav", audio_bytes)

        stt_start = time.perf_counter()
        stt_payload = await self.personaplex_client.stt(audio_bytes=audio_bytes, filename=input_filename)
        stt_ms = elapsed_ms(stt_start)

        transcript = (stt_payload.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("STT returned empty transcript")

        self.storage.save_json(job_id, "transcript.json", stt_payload)

        kb_start = time.perf_counter()
        kb_payload = await self.kb_client.search(query=transcript, top_k=top_k)
        kb_ms = elapsed_ms(kb_start)
        kb_results = normalize_kb_items(kb_payload)
        self.storage.save_json(job_id, "kb.json", {"raw": kb_payload, "normalized": kb_results})

        llm_start = time.perf_counter()
        response_text, citations = self._build_response_text(transcript=transcript, kb_results=kb_results)
        llm_ms = elapsed_ms(llm_start)

        tts_start = time.perf_counter()
        audio_out = await self.personaplex_client.tts(text=response_text, voice_prompt=voice_prompt)
        tts_ms = elapsed_ms(tts_start)
        self.storage.save_bytes(job_id, "output.wav", audio_out)

        response_payload = {
            "answer": response_text,
            "citations": citations,
        }
        self.storage.save_json(job_id, "response.json", response_payload)

        total_ms = elapsed_ms(overall_start)
        timings_ms = {
            "stt_ms": stt_ms,
            "kb_ms": kb_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
        }
        self.storage.save_json(job_id, "timings.json", timings_ms)
        self.storage.save_log(
            job_id,
            f"job_id={job_id} stt_ms={stt_ms} kb_ms={kb_ms} llm_ms={llm_ms} tts_ms={tts_ms} total_ms={total_ms}",
        )

        return PipelineResult(
            job_id=job_id,
            transcript=transcript,
            kb_results=kb_results,
            response_text=response_text,
            citations=citations,
            audio_bytes=audio_out,
            timings_ms=timings_ms,
        )

    async def run_from_text(
        self,
        text: str,
        voice_prompt: str,
        top_k: int,
    ) -> PipelineResult:
        overall_start = time.perf_counter()
        job_id = self.storage.create_job()

        transcript = text.strip()
        if not transcript:
            raise RuntimeError("Text input cannot be empty")

        self.storage.save_json(job_id, "transcript.json", {"text": transcript, "raw": {"source": "agent/text"}})

        kb_start = time.perf_counter()
        kb_payload = await self.kb_client.search(query=transcript, top_k=top_k)
        kb_ms = elapsed_ms(kb_start)
        kb_results = normalize_kb_items(kb_payload)
        self.storage.save_json(job_id, "kb.json", {"raw": kb_payload, "normalized": kb_results})

        llm_start = time.perf_counter()
        response_text, citations = self._build_response_text(transcript=transcript, kb_results=kb_results)
        llm_ms = elapsed_ms(llm_start)

        tts_start = time.perf_counter()
        audio_out = await self.personaplex_client.tts(text=response_text, voice_prompt=voice_prompt)
        tts_ms = elapsed_ms(tts_start)
        self.storage.save_bytes(job_id, "output.wav", audio_out)

        response_payload = {
            "answer": response_text,
            "citations": citations,
        }
        self.storage.save_json(job_id, "response.json", response_payload)

        timings_ms = {
            "stt_ms": 0,
            "kb_ms": kb_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": elapsed_ms(overall_start),
        }
        self.storage.save_json(job_id, "timings.json", timings_ms)
        self.storage.save_log(
            job_id,
            f"job_id={job_id} stt_ms=0 kb_ms={kb_ms} llm_ms={llm_ms} tts_ms={tts_ms} total_ms={timings_ms['total_ms']}",
        )

        return PipelineResult(
            job_id=job_id,
            transcript=transcript,
            kb_results=kb_results,
            response_text=response_text,
            citations=citations,
            audio_bytes=audio_out,
            timings_ms=timings_ms,
        )

    def _build_response_text(self, transcript: str, kb_results: list[dict[str, Any]]) -> tuple[str, list[str]]:
        snippets: list[str] = []
        citations: list[str] = []
        for item in kb_results[:5]:
            snippet = (
                item.get("text")
                or item.get("chunk")
                or item.get("content")
                or (item.get("metadata") or {}).get("text")
                or ""
            ).strip()
            source = (
                item.get("source")
                or (item.get("metadata") or {}).get("source")
                or (item.get("metadata") or {}).get("filename")
                or "unknown"
            )
            if snippet:
                snippets.append(snippet)
            citations.append(source)

        deduped_citations = list(dict.fromkeys(citations))

        if not snippets:
            return (
                "I could not find a direct policy match in our current knowledge base. "
                "Could you share more details so I can narrow this down?",
                deduped_citations,
            )

        kb_context = "\n\n".join(snippets)
        if self.settings.use_llm:
            llm_answer = self.openai_client.generate_answer(query=transcript, kb_context=kb_context)
            if not llm_answer:
                llm_answer = "I need a little more context to answer this accurately."
            return f"{llm_answer}\n\nSources: {', '.join(deduped_citations)}", deduped_citations

        summary = snippets[0]
        rule_based = (
            f"Based on HR policy, here is a concise answer: {summary} "
            f"If you need role-specific details, share your team and location. "
            f"Sources: {', '.join(deduped_citations)}"
        )
        return rule_based, deduped_citations


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
