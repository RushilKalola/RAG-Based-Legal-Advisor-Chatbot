import asyncio
from app.utils.config import settings
from app.services.retrieval import RetrievalService

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# Global semaphore: only N requests call Mistral at the same time
_mistral_semaphore = asyncio.Semaphore(3)

class ChatService:
    def __init__(self):
        self.retrieval = RetrievalService()
        self.client = ChatMistralAI(
            api_key=settings.MISTRAL_API_KEY,
            model=settings.MISTRAL_MODEL,
            max_retries=6,  # LangChain handles retry + backoff internally
        )

    async def get_answer(self, query: str):
        # Run blocking search in thread
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            self.retrieval.search,
            query
        )

        context = "\n\n".join([doc["text"] for doc in results])

        prompt = f"""You are a legal assistant.
Answer ONLY using the context provided below.
Do not use any external knowledge or make assumptions beyond what is written.
If the answer is not in the context, say "I could not find this in the provided legal documents."
Context:
{context}

Question:
{query}

Answer clearly and cite the relevant section if possible.
"""

        messages = [HumanMessage(content=prompt)]

        # Throttle: wait here until a slot is free before calling Mistral
        async with _mistral_semaphore:
            response = await self.client.ainvoke(messages)

        answer = response.content

        return {
            "answer": answer,
            "sources": [doc["source"] for doc in results]
        }