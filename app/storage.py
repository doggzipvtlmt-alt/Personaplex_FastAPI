import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JobStorage:
    def __init__(self, output_dir: str):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, initial_meta: dict[str, Any] | None = None) -> str:
        job_id = uuid.uuid4().hex
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "job_id": job_id,
            "status": "queued",
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        if initial_meta:
            meta.update(initial_meta)

        self.save_json(job_id, "job.json", meta)
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def job_exists(self, job_id: str) -> bool:
        return self.job_dir(job_id).exists()

    def save_bytes(self, job_id: str, filename: str, data: bytes) -> Path:
        path = self.job_dir(job_id) / filename
        path.write_bytes(data)
        return path

    def save_text(self, job_id: str, filename: str, text: str) -> Path:
        path = self.job_dir(job_id) / filename
        path.write_text(text, encoding="utf-8")
        return path

    def save_json(self, job_id: str, filename: str, payload: dict[str, Any]) -> Path:
        path = self.job_dir(job_id) / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_json(self, job_id: str, filename: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / filename
        return json.loads(path.read_text(encoding="utf-8"))

    def read_text(self, job_id: str, filename: str) -> str:
        path = self.job_dir(job_id) / filename
        return path.read_text(encoding="utf-8")

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        meta = self.read_json(job_id, "job.json")
        meta.update(updates)
        meta["updated_at"] = self._now()
        self.save_json(job_id, "job.json", meta)
        return meta

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.read_json(job_id, "job.json")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
