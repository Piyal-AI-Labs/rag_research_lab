import faiss
import numpy as np

class FAISSStore:

    def __init__(
        self,
        dimension: int
    ):
        
        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.documents = []


    def add_documents(
        self,
        embeddings,
        documents
    ):
        
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        
        self.documents.extend(documents)


    def search(
        self,
        query_embedding,
        k=5
    ):
        
        query = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(query, k)

        results = []

        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results