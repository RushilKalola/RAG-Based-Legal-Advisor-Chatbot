from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.tools.chat_tool import ChatTool

router = APIRouter()

chat_tool = ChatTool()


# Request schema
class ChatRequest(BaseModel):
    query: str


# Response schema
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = await chat_tool.ask(request.query)

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))