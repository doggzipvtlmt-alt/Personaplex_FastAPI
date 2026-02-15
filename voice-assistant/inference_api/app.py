from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .kb import kb, router as kb_router

APP_ROOT = Path('/workspace/voice-assistant')
UPLOAD_DIR = APP_ROOT / 'uploads'
OUTPUT_DIR = APP_ROOT / 'outputs'
LOG_DIR = APP_ROOT / 'logs'
PERSONAPLEX_ROOT = Path('/workspace/personaplex')
DEFAULT_VOICE_PROMPT = 'NATF2.pt'


def ensure_dirs() -> None:
    for directory in [APP_ROOT, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(title='RunPod PersonaPlex FastAPI', version='1.0.0')
app.include_router(kb_router)


@app.on_event('startup')
def startup() -> None:
    ensure_dirs()
    kb.load()


@app.get('/health')
def health() -> dict:
    return {
        'ok': True,
        'hf_token_set': bool(os.getenv('HF_TOKEN')),
        'personaplex_root_exists': PERSONAPLEX_ROOT.exists(),
        'uploads_dir': str(UPLOAD_DIR),
        'outputs_dir': str(OUTPUT_DIR),
    }


@app.post('/v1/offline')
async def offline_inference(
    file: UploadFile = File(...),
    voice_prompt: str = Query(DEFAULT_VOICE_PROMPT),
    seed: int = Query(1234),
) -> dict:
    if not os.getenv('HF_TOKEN'):
        raise HTTPException(status_code=500, detail='HF_TOKEN is not set')

    if not file.filename or not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail='Only .wav file uploads are supported')

    ensure_dirs()
    job_id = uuid.uuid4().hex
    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    upload_path = UPLOAD_DIR / f'{job_id}_{Path(file.filename).name}'
    input_bytes = await file.read()
    upload_path.write_bytes(input_bytes)

    output_wav = job_output_dir / 'output.wav'
    output_json = job_output_dir / 'output.json'
    run_log = job_output_dir / 'run.log'

    command = [
        'python',
        '-m',
        'moshi.offline',
        '--voice-prompt',
        voice_prompt,
        '--input-wav',
        str(upload_path),
        '--seed',
        str(seed),
        '--output-wav',
        str(output_wav),
        '--output-text',
        str(output_json),
    ]

    completed = subprocess.run(
        command,
        cwd=str(PERSONAPLEX_ROOT),
        capture_output=True,
        text=True,
    )

    combined_log = (
        f'COMMAND: {" ".join(command)}\n\n'
        f'RETURN_CODE: {completed.returncode}\n\n'
        f'STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n'
    )
    run_log.write_text(combined_log, encoding='utf-8')

    if completed.returncode != 0:
        tail_lines = (completed.stderr or completed.stdout).splitlines()[-40:]
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Inference failed',
                'job_id': job_id,
                'log_tail': '\n'.join(tail_lines),
                'run_log': str(run_log),
            },
        )

    if not output_wav.exists() or not output_json.exists():
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Inference completed but outputs are missing',
                'job_id': job_id,
                'run_log': str(run_log),
            },
        )

    return {
        'job_id': job_id,
        'output_audio_url': f'/v1/result/{job_id}/audio',
        'output_json_url': f'/v1/result/{job_id}/json',
        'run_log': str(run_log),
    }


@app.get('/v1/result/{job_id}/audio')
def get_result_audio(job_id: str) -> FileResponse:
    output_wav = OUTPUT_DIR / job_id / 'output.wav'
    if not output_wav.exists():
        raise HTTPException(status_code=404, detail='Output audio not found')
    return FileResponse(path=output_wav, media_type='audio/wav', filename='output.wav')


@app.get('/v1/result/{job_id}/json')
def get_result_json(job_id: str) -> dict:
    output_json = OUTPUT_DIR / job_id / 'output.json'
    if not output_json.exists():
        raise HTTPException(status_code=404, detail='Output json not found')
    try:
        return json.loads(output_json.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'raw_text': output_json.read_text(encoding='utf-8', errors='ignore')}
