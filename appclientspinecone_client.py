import os
from pinecone import Pinecone

class PineconeClient:

    def __init__(self):

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        host = os.getenv("PINECONE_HOST")
        namespace = os.getenv("PINECONE_NAMESPACE", "xccelera-hr")

        pc = Pinecone(api_key=api_key)

        self.index = pc.Index(
            name=index_name,
            host=host,
        )

        self.namespace = namespace


    def query(self, vector, top_k=5):

        result = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        return result