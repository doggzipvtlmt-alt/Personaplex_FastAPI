from pathlib import Path
from typing import List, Dict

# Your KB folders are at repo root: /faq /hiring /policies ...
REPO_ROOT = Path(__file__).resolve().parent.parent

KB_FOLDERS = ["faq", "hiring", "policies"]

def iter_md_files():
    for folder in KB_FOLDERS:
        p = REPO_ROOT / folder
        if p.exists() and p.is_dir():
            yield from p.rglob("*.md")

def load_docs() -> List[Dict]:
    docs = []
    for f in iter_md_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            {
                "id": str(f.relative_to(REPO_ROOT)).replace("\\", "/"),
                "text": text,
            }
        )
    return docs

def keyword_search(query: str, k: int = 3) -> List[Dict]:
    q = (query or "").lower().strip()
    if not q:
        return []

    tokens = [t for t in q.split() if t]
    docs = load_docs()

    scored = []
    for d in docs:
        t = d["text"].lower()
        score = sum(1 for tok in tokens if tok in t)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]
