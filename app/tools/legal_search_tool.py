from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from app.utils.config import settings


class LegalSearchTool:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60
        )

    def search(self, query: str):
        # Step 1: Convert query to embedding
        query_vector = self.embedding_model.encode(query).tolist()

        # Step 2: Search in Qdrant
        results = self.qdrant.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=settings.TOP_K_RESULTS,
        score_threshold=settings.SCORE_THRESHOLD,
        search_params={"hnsw_ef": 256},  
    )

        # Step 3: Format results
        documents = []
        for res in results:
            documents.append({
                "text": res.payload.get("text", ""),
                "source": res.payload.get("source", ""),
                "score": res.score
            })

        return documents