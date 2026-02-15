from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

KB_ROOT = Path('/workspace/knowledge_base')


@dataclass
class KBDoc:
    doc_id: str
    path: str
    content: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class KnowledgeBase:
    def __init__(self, kb_root: Path = KB_ROOT) -> None:
        self.kb_root = kb_root
        self.docs: List[KBDoc] = []
        self.error: str | None = None

    def load(self) -> None:
        self.docs = []
        self.error = None

        if not self.kb_root.exists() or not self.kb_root.is_dir():
            self.error = f'Knowledge base folder not found: {self.kb_root}'
            return

        for md_file in sorted(self.kb_root.rglob('*.md')):
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                relative = md_file.relative_to(self.kb_root)
                self.docs.append(
                    KBDoc(doc_id=str(relative), path=str(md_file), content=text)
                )
            except Exception as exc:  # noqa: BLE001
                self.error = f'Failed to load {md_file}: {exc}'

    def health(self) -> dict:
        return {
            'ok': self.error is None,
            'kb_root': str(self.kb_root),
            'doc_count': len(self.docs),
            'error': self.error,
        }

    @staticmethod
    def _score(content: str, query: str) -> int:
        tokens = [t.lower() for t in query.split() if t.strip()]
        content_l = content.lower()
        return sum(content_l.count(token) for token in tokens)

    def search(self, query: str, top_k: int) -> list[dict]:
        if self.error:
            raise HTTPException(status_code=500, detail=self.error)

        scores = []
        for doc in self.docs:
            score = self._score(doc.content, query)
            if score > 0:
                snippet = doc.content[:500].replace('\n', ' ').strip()
                scores.append(
                    {
                        'doc_id': doc.doc_id,
                        'path': doc.path,
                        'score': score,
                        'snippet': snippet,
                    }
                )

        scores.sort(key=lambda item: item['score'], reverse=True)
        return scores[:top_k]


kb = KnowledgeBase()
router = APIRouter(prefix='/kb', tags=['knowledge-base'])


@router.get('/health')
def kb_health() -> dict:
    return kb.health()


@router.get('/docs')
def kb_docs() -> dict:
    if kb.error:
        raise HTTPException(status_code=500, detail=kb.error)
    return {
        'count': len(kb.docs),
        'docs': [{'doc_id': d.doc_id, 'path': d.path} for d in kb.docs],
    }


@router.post('/search')
def kb_search(payload: SearchRequest) -> dict:
    return {
        'query': payload.query,
        'top_k': payload.top_k,
        'results': kb.search(payload.query, payload.top_k),
    }
