from fastapi import APIRouter, Query

from app.clients.pinecone_client import PineconeClient
from app.services.embedding import embed_text

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

pc = PineconeClient()


@router.get("/search")
def search_kb(
    q: str = Query(...),
    k: int = 5,
):

    vector = embed_text(q)

    result = pc.query(
        vector=vector,
        top_k=k,
    )

    matches = []

    for m in result.matches:

        matches.append(
            {
                "id": m.id,
                "score": m.score,
                "text": m.metadata.get("text"),
                "source": m.metadata.get("source"),
            }
        )

    return {
        "query": q,
        "results": matches,
    }