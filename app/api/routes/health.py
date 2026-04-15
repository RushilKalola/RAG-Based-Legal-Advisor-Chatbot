from fastapi import APIRouter
from app.utils.config import settings

router = APIRouter()


@router.get("/")
def health_check():
    return {
        "status": "ok",
        "environment": settings.APP_ENV,
        "app_name": "RAG Legal Advisor Chatbot"
    }