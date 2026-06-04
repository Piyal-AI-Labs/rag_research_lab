from .base import  BaseRetriever

class VectorRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store,
        embedding_model,
        top_k=5
    ):
        
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(
        self, 
        query: str
    ):
        
        query_embedding = self.embedding_model.embed_text(query)

        return self.vector_store.search(
            query_embedding,
            self.top_k
        )