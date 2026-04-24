from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, compare
from app.utils.config import settings

app = FastAPI(
    title="RAG Legal Advisor Chatbot",
    description="AI-powered legal assistant using RAG + Qdrant + Mistral",
    version="1.0.0"
)

# CORS (important for frontend like Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(compare.router, prefix="/compare", tags=["Compare"])

@app.get("/")
def root():
    return {
        "message": "RAG Legal Advisor Chatbot is running 🚀",
        "environment": settings.APP_ENV
        
    }

