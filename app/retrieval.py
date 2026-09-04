import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import BM25_TOP_K, VECTOR_TOP_K


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            document.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=BM25_TOP_K):
        scores = self.bm25.get_scores(query.lower().split())

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self.documents[index],
                "score": float(scores[index]),
                "method": "bm25"
            }
            for index in top_indices
        ]


class VectorRetriever:
    def __init__(self, documents):
        self.documents = documents

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype="float32"))

    def search(self, query, top_k=VECTOR_TOP_K):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype="float32"),
            top_k
        )

        return [
            {
                "text": self.documents[index],
                "score": float(scores[0][position]),
                "method": "vector"
            }
            for position, index in enumerate(indices[0])
            if index >= 0
        ]


class HybridRetriever:
    def __init__(self, documents):
        self.bm25 = BM25Retriever(documents)
        self.vector = VectorRetriever(documents)

    def search(self, query):
        bm25_results = self.bm25.search(query)
        vector_results = self.vector.search(query)

        combined = {}

        for result in bm25_results + vector_results:
            text = result["text"]

            if text not in combined:
                combined[text] = {
                    "text": text,
                    "score": 0.0,
                    "methods": []
                }

            combined[text]["score"] += result["score"]

            if result["method"] not in combined[text]["methods"]:
                combined[text]["methods"].append(result["method"])

        return sorted(
            combined.values(),
            key=lambda item: item["score"],
            reverse=True
        )