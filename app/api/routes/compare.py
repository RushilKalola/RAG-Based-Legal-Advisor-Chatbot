from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.tools.act_comparison_tool import ActComparisonTool

router = APIRouter()
comparison_tool = ActComparisonTool()


class CompareRequest(BaseModel):
    topic: str      # e.g. "punishment for theft"
    act_a: str      # e.g. "Bharatiya Nyaya Sanhita"
    act_b: str      # e.g. "Indian Penal Code"


class CompareResponse(BaseModel):
    topic: str
    act_a: str
    act_b: str
    comparison: str
    sources_a: list[str]
    sources_b: list[str]


@router.post("/", response_model=CompareResponse)
async def compare_acts(request: CompareRequest):
    """
    Fetch relevant chunks from two acts simultaneously and produce a
    structured side-by-side comparison on the given topic.
    """
    try:
        result = await comparison_tool.compare(
            topic=request.topic,
            act_a=request.act_a,
            act_b=request.act_b,
        )
        return CompareResponse(**result)

    except RuntimeError as e:
        if "rate limit" in str(e).lower() or "max retries" in str(e).lower():
            raise HTTPException(
                status_code=429,
                detail="The AI service is busy. Please wait a moment and try again."
            )
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))