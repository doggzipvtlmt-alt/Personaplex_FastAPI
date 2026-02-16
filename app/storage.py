import json
import uuid
from pathlib import Path
from typing import Any


class JobStorage:
    def __init__(self, output_dir: str):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self) -> str:
        job_id = uuid.uuid4().hex
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def save_bytes(self, job_id: str, filename: str, data: bytes) -> Path:
        path = self.job_dir(job_id) / filename
        path.write_bytes(data)
        return path

    def save_json(self, job_id: str, filename: str, payload: dict[str, Any]) -> Path:
        path = self.job_dir(job_id) / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_log(self, job_id: str, line: str) -> Path:
        path = self.job_dir(job_id) / "run.log"
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
        return path

    def read_json(self, job_id: str, filename: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / filename
        return json.loads(path.read_text(encoding="utf-8"))

    def read_audio(self, job_id: str, filename: str = "output.wav") -> bytes:
        path = self.job_dir(job_id) / filename
        return path.read_bytes()
