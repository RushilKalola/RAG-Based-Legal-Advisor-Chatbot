import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings:
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 768))

    # # Chunking
    # CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    # CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # LLM (Mistral)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL")

    # FastAPI
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 10))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", 0.50))


# Create a global settings object
settings = Settings()