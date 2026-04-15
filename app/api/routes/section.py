from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.section_services import SectionService

router = APIRouter()

section_service = SectionService()


class SectionRequest(BaseModel):
    query: str  # e.g. "Article 21 Constitution" or "IPC Section 302"


class SectionResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


@router.post("/", response_model=SectionResponse)
async def section_lookup(request: SectionRequest):
    """
    Look up a specific legal section and return the exact verbatim text from the document.
    Uses SectionService with a strict verbatim-extraction prompt.
    """
    try:
        result = await section_service.get_section(request.query)

        return SectionResponse(
            query=request.query,
            answer=result["answer"],
            sources=result.get("sources", [])
        )

    except RuntimeError as e:
        if "rate limit" in str(e).lower() or "max retries" in str(e).lower():
            raise HTTPException(
                status_code=429,
                detail="The AI service is busy. Please wait a moment and try again."
            )
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))