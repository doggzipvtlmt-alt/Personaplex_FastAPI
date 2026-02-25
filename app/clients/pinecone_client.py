import os
from typing import Any


class PineconeClient:
    def __init__(self) -> None:
        self.namespace = os.getenv("PINECONE_NAMESPACE", "xccelera-hr")
        self._index = None

    def _get_index(self):
        if self._index is not None:
            return self._index

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        host = os.getenv("PINECONE_HOST")

        if not api_key:
            raise RuntimeError("Missing PINECONE_API_KEY")
        if not index_name:
            raise RuntimeError("Missing PINECONE_INDEX")
        if not host:
            raise RuntimeError("Missing PINECONE_HOST")

        try:
            from pinecone import Pinecone
        except ModuleNotFoundError as exc:
            raise RuntimeError("pinecone package is not installed") from exc

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(name=index_name, host=host)
        return self._index

    def query(self, vector: list[float], top_k: int = 5) -> Any:
        index = self._get_index()
        return index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
