import asyncio
from pathlib import Path
from app.kb import search_kb
import httpx

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.clients import OpenAIClient, PersonaPlexClient
from app.config import get_settings
from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService
from app.storage import JobStorage

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".m4a"}

settings = get_settings()
storage = JobStorage(output_dir=settings.output_dir)
#kb_client = KBClient(settings=settings)
openai_client = OpenAIClient(settings=settings)
personaplex_client = PersonaPlexClient(settings=settings)
stt_service = STTService(settings=settings)
tts_service = TTSService(settings=settings)
llm_service = LLMService(openai_client=openai_client, personaplex_client=personaplex_client)

templates = Jinja2Templates(directory="templates")

app = FastAPI(title=settings.app_name, version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

if settings.environment.lower() == "production":
    allow_origins = settings.allowed_origins
else:
    allow_origins = list(dict.fromkeys([*settings.allowed_origins, "http://localhost", "http://127.0.0.1"]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}

@app.get("/kb/search")
def kb_search(q: str = Query(...), k: int = 3):
    results = search_kb(q, k=k)

    # keep response light
    formatted = [
        {
            "id": r["id"],
            "preview": r["text"][:800],   # first 800 chars only
        }
        for r in results
    ]

    return {"query": q, "k": k, "results_count": len(results), "results": formatted}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/debug/kb")
def debug_kb(q: str = Query(...), k: int = 3):
    # Local repo markdown search
    results = search_kb(q, k=k)
    return {
        "query": q,
        "k": k,
        "results_count": len(results),
        "results": [{"id": r["id"], "preview": r["text"][:500]} for r in results],
    }


@app.post("/api/voice")
async def api_voice(
    file: UploadFile = File(...),
    mode: str = Form(default="default"),
    user_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
):
    _validate_upload(file)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file too large")

    extension = Path(file.filename or "recording.webm").suffix.lower() or ".webm"
    job_id = storage.create_job(
        {
            "status": "queued",
            "mode": mode,
            "user_id": user_id,
            "session_id": session_id,
            "filename": file.filename,
        }
    )

    storage.update_job(job_id, status="processing")

    audio_input_name = f"input{extension}"
    audio_input_path = storage.save_bytes(job_id, audio_input_name, audio_bytes)

    try:
        transcript = await stt_service.transcribe(audio_input_path)

        # Use local markdown KB instead of external KB client
        kb_results = search_kb(transcript, k=5)
        
        citations = [r["id"] for r in kb_results]
        
        kb_context = "\n\n".join(
            f"## {r['id']}\n{r['text'][:1500]}"
            for r in kb_results
        )

        assistant_text = await llm_service.generate(transcript=transcript, context=kb_context)
        output_extension = ".mp3" if settings.elevenlabs_api_key else ".wav"
        output_path = storage.job_dir(job_id) / f"output{output_extension}"
        final_audio_path = await tts_service.synthesize(text=assistant_text, output_path=output_path)

        storage.save_text(job_id, "transcript.txt", transcript)
        storage.save_text(job_id, "response.txt", assistant_text)
        storage.save_json(
            job_id,
            "meta.json",
            {
                "job_id": job_id,
                "transcript": transcript,
                "assistant_text": assistant_text,
                "citations": citations,
                "audio_file": final_audio_path.name,
            },
        )
        storage.update_job(job_id, status="completed", audio_file=final_audio_path.name)

    except Exception as exc:
        storage.update_job(job_id, status="failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {exc}") from exc

    payload = {
        "job_id": job_id,
        "status": "completed",
        "transcript": transcript,
        "assistant_text": assistant_text,
    }
    return JSONResponse(content=payload)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    _ensure_job_exists(job_id)
    return JSONResponse(content=storage.get_job(job_id))


@app.get("/api/jobs/{job_id}/audio")
async def get_job_audio(job_id: str):
    _ensure_job_exists(job_id)
    meta = storage.get_job(job_id)
    audio_file = meta.get("audio_file", "output.wav")
    audio_path = storage.job_dir(job_id) / audio_file
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    media_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path=audio_path, media_type=media_type, filename=audio_path.name)


@app.get("/api/jobs/{job_id}/meta")
async def get_job_meta(job_id: str):
    _ensure_job_exists(job_id)
    meta_path = storage.job_dir(job_id) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not ready")
    return JSONResponse(content=storage.read_json(job_id, "meta.json"))


@app.get("/api/jobs/{job_id}/events")
async def get_job_events(job_id: str):
    _ensure_job_exists(job_id)

    async def event_stream():
        last_status = None
        for _ in range(30):
            status = storage.get_job(job_id).get("status", "unknown")
            if status != last_status:
                yield f"data: {status}\n\n"
                last_status = status
            if status in {"completed", "failed"}:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _validate_upload(file: UploadFile) -> None:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )


def _extract_citations(kb_results: list[dict]) -> list[str]:
    citations: list[str] = []
    for item in kb_results:
        source = (
            item.get("source")
            or (item.get("metadata") or {}).get("source")
            or (item.get("metadata") or {}).get("filename")
        )
        if source:
            citations.append(str(source))
    return list(dict.fromkeys(citations))


def _ensure_job_exists(job_id: str) -> None:
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/debug/personaplex")
async def debug_personaplex():
    url = settings.personaplex_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        return {
            "ok": True,
            "personaplex_health_url": url,
            "status": r.status_code,
            "body": r.json()
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PersonaPlex not reachable: {exc}"
        )

@app.get("/debug/kb")
def debug_kb(q: str = Query(...), k: int = 3):
    results = search_kb(q, k=k)
    return {
        "query": q,
        "k": k,
        "results_count": len(results),
        "results": results,
    }
