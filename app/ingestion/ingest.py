from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.utils.config import settings
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_splitter import TextSplitter
from app.utils.logger import log

import uuid
import os


class IngestionPipeline:
    def __init__(self):
        self.loader = PDFLoader()
        self.splitter = TextSplitter()

        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if settings.QDRANT_COLLECTION_NAME not in collection_names:
            log.info(f"Creating collection: {settings.QDRANT_COLLECTION_NAME}")

            self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
        else:
            log.info("Collection already exists. Skipping creation.")

    def ingest(self, file_path: str):
        log.info(f"Processing file: {file_path}")

        # Step 1: Load PDF
        text = self.loader.load(file_path)

        if not text.strip():
            log.warning(f"No text found in {file_path}")
            return {"status": "failed", "reason": "empty document"}

        # Step 2: Split text
        chunks = self.splitter.split(text)
        log.info(f"Total chunks created: {len(chunks)}")

        # Step 3: Create embeddings
        vectors = self.embedding_model.encode(chunks)

        # Step 4: Prepare points (batch)
        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[i].tolist(),
                    payload={
                        "text": chunk,
                        "source": os.path.basename(file_path)
                    }
                )
            )

        # Step 5: Upload in batches (important for large files)
        BATCH_SIZE = 25

        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]

            self.qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=batch
            )

        log.info(f"Ingestion complete: {file_path}")

        return {
            "status": "success",
            "file": file_path,
            "chunks_indexed": len(points)
        }

# ✅ Runner (so you can execute directly)
if __name__ == "__main__":
    pipeline = IngestionPipeline()

    data_path = "data/raw/"

    if not os.path.exists(data_path):
        print("❌ data/raw/ folder not found")
        exit()

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_path, file)

            result = pipeline.ingest(file_path)
            print(result)

    print("\n✅ All files processed successfully!")