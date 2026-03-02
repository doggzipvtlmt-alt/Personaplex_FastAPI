from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from app.config import get_settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]

try:
    from pinecone import Pinecone
except Exception:  # pragma: no cover - optional dependency at runtime
    Pinecone = None  # type: ignore[assignment]

# Your KB folders are at repo root: /faq /hiring /policies ...
REPO_ROOT = Path(__file__).resolve().parent.parent
KB_FOLDERS = ["faq", "hiring", "policies"]


def iter_md_files():
    for folder in KB_FOLDERS:
        p = REPO_ROOT / folder
        if p.exists() and p.is_dir():
            yield from p.rglob("*.md")


def load_docs() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for f in iter_md_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            {
                "id": str(f.relative_to(REPO_ROOT)).replace("\\", "/"),
                "text": text,
            }
        )
    return docs


def keyword_search(query: str, k: int = 3) -> List[Dict[str, str]]:
    q = (query or "").lower().strip()
    if not q:
        return []

    tokens = [t for t in q.split() if t]
    docs = load_docs()

    scored: list[tuple[int, Dict[str, str]]] = []
    for d in docs:
        t = d["text"].lower()
        score = sum(1 for tok in tokens if tok in t)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


@lru_cache(maxsize=1)
def _pinecone_index():
    settings = get_settings()
    if not settings.pinecone_api_key or not settings.pinecone_index_name or Pinecone is None:
        return None
    client = Pinecone(api_key=settings.pinecone_api_key)
    return client.Index(settings.pinecone_index_name)


def _embed_query(query: str) -> list[float] | None:
    settings = get_settings()
    if not settings.openai_api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=settings.openai_api_key)
    embedding = client.embeddings.create(
        model=settings.embedding_model,
        input=query,
    )
    return embedding.data[0].embedding


def pinecone_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    index = _pinecone_index()
    vector = _embed_query(query)
    if index is None or vector is None:
        return []

    settings = get_settings()
    response = index.query(
        vector=vector,
        top_k=k,
        include_metadata=True,
        namespace=settings.pinecone_namespace or None,
    )

    results: List[Dict[str, Any]] = []
    for match in getattr(response, "matches", []) or []:
        metadata = dict(match.metadata or {})
        text = str(metadata.get("text") or metadata.get("content") or "")
        doc_id = str(metadata.get("id") or metadata.get("source") or match.id)
        if not text:
            continue
        results.append(
            {
                "id": doc_id,
                "text": text,
                "score": float(match.score or 0.0),
                "metadata": metadata,
            }
        )
    return results


def search_kb(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search via Pinecone with safe fallback to local keyword search."""
    q = (query or "").strip()
    if not q:
        return []

    try:
        semantic = pinecone_search(q, k=k)
        if semantic:
            return semantic
    except Exception:
        # Keep gateway resilient for voice path; fallback to existing local keyword search.
        pass

    return keyword_search(q, k=k)
