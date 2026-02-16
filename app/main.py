from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.clients import KBClient, OpenAIClient, PersonaPlexClient
from app.config import get_settings
from app.models import AgentUrlResponse, HealthResponse, JobJsonResponse, ResponsePayload, TextAgentRequest
from app.pipeline import VoicePipeline
from app.storage import JobStorage

settings = get_settings()
storage = JobStorage(output_dir=settings.output_dir)
personaplex_client = PersonaPlexClient(settings=settings)
kb_client = KBClient(settings=settings)
openai_client = OpenAIClient(settings=settings)
pipeline = VoicePipeline(
    settings=settings,
    storage=storage,
    personaplex_client=personaplex_client,
    kb_client=kb_client,
    openai_client=openai_client,
)

app = FastAPI(title="PersonaPlex Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/agent/voice")
async def agent_voice(
    file: UploadFile = File(...),
    voice_prompt: str = Form(default="NATF2.pt"),
    top_k: int = Form(default=5),
    return_mode: str = Form(default="file"),
):
    if return_mode not in {"file", "url"}:
        raise HTTPException(status_code=400, detail="return_mode must be file or url")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be in range [1, 20]")

    if file.content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only WAV uploads are supported")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file too large")

    if not (audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid WAV")

    try:
        result = await pipeline.run_from_audio(
            audio_bytes=audio_bytes,
            voice_prompt=voice_prompt,
            top_k=top_k,
            input_filename=file.filename or "input.wav",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if return_mode == "file":
        return Response(content=result.audio_bytes, media_type="audio/wav")

    audio_url = f"/results/{result.job_id}/audio"
    payload = AgentUrlResponse(
        job_id=result.job_id,
        audio_url=audio_url,
        transcript=result.transcript,
        kb_sources=result.citations,
    )
    return JSONResponse(content=payload.model_dump())


@app.post("/agent/text")
async def agent_text(request: TextAgentRequest):
    try:
        result = await pipeline.run_from_text(
            text=request.text,
            voice_prompt=request.voice_prompt,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if request.return_mode == "file":
        return Response(content=result.audio_bytes, media_type="audio/wav")

    audio_url = f"/results/{result.job_id}/audio"
    payload = AgentUrlResponse(
        job_id=result.job_id,
        audio_url=audio_url,
        transcript=result.transcript,
        kb_sources=result.citations,
    )
    return JSONResponse(content=payload.model_dump())


@app.get("/results/{job_id}/audio")
async def get_audio(job_id: str):
    audio_path = Path(settings.output_dir) / job_id / "output.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio artifact not found")
    return FileResponse(path=audio_path, media_type="audio/wav", filename="output.wav")


@app.get("/results/{job_id}/json", response_model=JobJsonResponse)
async def get_json(job_id: str):
    job_path = Path(settings.output_dir) / job_id
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    transcript_payload = storage.read_json(job_id, "transcript.json")
    kb_payload = storage.read_json(job_id, "kb.json")
    response_payload = storage.read_json(job_id, "response.json")
    timings_payload = storage.read_json(job_id, "timings.json")

    result = JobJsonResponse(
        job_id=job_id,
        transcript=transcript_payload.get("text", ""),
        kb_results=kb_payload.get("normalized", []),
        response=ResponsePayload(**response_payload),
        timings_ms=timings_payload,
    )
    return JSONResponse(content=result.model_dump())
