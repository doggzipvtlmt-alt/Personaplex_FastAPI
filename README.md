# PersonaPlex FastAPI Voice Assistant

A production-friendly FastAPI web app that records audio in the browser, runs STT + KB + LLM + TTS, and returns playable assistant audio.

## Features

- `GET /health` health check
- `GET /` web UI with microphone recording
- `POST /api/voice` audio upload + full pipeline
- `GET /api/jobs/{job_id}` job status
- `GET /api/jobs/{job_id}/audio` generated audio
- `GET /api/jobs/{job_id}/meta` transcript + assistant text + citations
- `GET /api/jobs/{job_id}/events` SSE status updates
- Works with ElevenLabs/OpenAI when configured, includes no-secret fallback mode

## Environment variables

- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `ELEVENLABS_API_KEY` (optional)
- `ELEVENLABS_VOICE_ID` (optional)
- `KB_BASE_URL` (optional)
- `KB_API_KEY` (optional)
- `PERSONAPLEX_BASE_URL` (optional)
- `PINECONE_API_KEY` (optional, enables semantic KB search)
- `PINECONE_INDEX_NAME` (optional, enables semantic KB search)
- `PINECONE_NAMESPACE` (optional)
- `EMBEDDING_MODEL` (optional, default `text-embedding-3-small`)
- `ALLOWED_ORIGINS` (optional list)
- `OUTPUT_DIR` (optional, default `/tmp/personaplex_outputs`)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: <http://localhost:8000>

## Render deployment

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT`

Set env vars in Render dashboard (only secrets you need). If Pinecone variables are not provided, the gateway automatically falls back to local markdown keyword search so PersonaPlex/STT/TTS flow keeps working.

## Minimal validation

```bash
curl -s http://localhost:8000/health
```

Then open `/`, record audio, and verify transcript/response/audio. You can inspect metadata:

```bash
curl -s http://localhost:8000/api/jobs/<job_id>/meta
```
