import os
from pinecone import Pinecone
 
 
class PineconeClient:
 
    def __init__(self):
 
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        host = os.getenv("PINECONE_HOST")
        namespace = os.getenv("PINECONE_NAMESPACE", "xccelera-hr")
 
        if not api_key:
            raise RuntimeError("Missing PINECONE_API_KEY")
 
        if not index_name:
            raise RuntimeError("Missing PINECONE_INDEX")
 
        if not host:
            raise RuntimeError("Missing PINECONE_HOST")
 
        self.namespace = namespace
 
        pc = Pinecone(api_key=api_key)
 
        self.index = pc.Index(
            name=index_name,
            host=host,
        )
 
 
    def query(self, vector, top_k=5):
 
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
 
        return result