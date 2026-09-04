from sentence_transformers import CrossEncoder

from app.config import RERANK_TOP_K


class CrossEncoderReranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, results, top_k=RERANK_TOP_K):
        if not results:
            return []

        pairs = [
            [query, result["text"]]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            reranked.append({
                **result,
                "rerank_score": float(score)
            })

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]