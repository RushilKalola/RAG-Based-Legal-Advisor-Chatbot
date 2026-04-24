from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

from app.utils.config import settings

class RetrievalService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Cross-encoder reranker — runs after Qdrant, re-scores each chunk
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

        self.qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60
        )

    def search(self, query: str):
        # Step 1: Convert query to embedding
        query_vector = self.embedding_model.encode(query).tolist()

        # Step 2: Broad retrieval from Qdrant (fetch more than you need)
        results = self.qdrant.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=settings.TOP_K_RESULTS,       # still 12 from .env
            score_threshold=settings.SCORE_THRESHOLD,
            search_params={"hnsw_ef": 256},
        )

        # Step 3: Rerank — cross-encoder scores each (query, chunk) pair
        candidates = [
            {
                "text": res.payload.get("text", ""),
                "source": res.payload.get("source", ""),
                "vector_score": res.score,
            }
            for res in results
        ]

        if candidates:
            pairs = [[query, doc["text"]] for doc in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for doc, score in zip(candidates, rerank_scores):
                doc["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Step 4: Return only top 5 after reranking
        top_k = int(settings.RERANK_TOP_K) if hasattr(settings, "RERANK_TOP_K") else 5
        documents = []
        for doc in candidates[:top_k]:
            documents.append({
                "text": doc["text"],
                "source": doc["source"],
                "score": doc.get("rerank_score", doc["vector_score"]),
            })

        return documents