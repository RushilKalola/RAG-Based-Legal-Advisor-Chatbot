import asyncio
import time
import random
from concurrent.futures import ThreadPoolExecutor

from app.utils.config import settings
from app.tools.legal_search_tool import LegalSearchTool

from mistralai import Mistral
from mistralai import models as mistral_models

# Import the shared semaphore from chat_services so both services
# share the same gate against Mistral's rate limit
from app.services.chat_services import _mistral_semaphore


class SectionService:
    def __init__(self):
        self.search_tool = LegalSearchTool()
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.executor = ThreadPoolExecutor()

    def _call_mistral_with_retry(self, model: str, messages: list, max_retries: int = 6) -> object:
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                return self.client.chat.complete(model=model, messages=messages)

            except mistral_models.SDKError as exc:
                is_rate_limit = (
                    "429" in str(exc)
                    or "rate_limit" in str(exc).lower()
                    or "rate limited" in str(exc).lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt) + random.uniform(0.0, 1.0)
                    print(
                        f"    [Mistral] Rate limited (429). "
                        f"Retrying in {wait:.1f}s… "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Mistral API: max retries exceeded due to rate limiting.")

    async def get_section(self, query: str):
        loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            self.executor,
            self.search_tool.search,
            query
        )

        if not results:
            return {
                "answer": "No matching section found in the provided legal documents.",
                "sources": []
            }

        context = "\n\n".join([doc["text"] for doc in results])

        prompt = f"""You are a legal document retrieval assistant.
Your ONLY job is to find and return the text of the requested section from the context below.

And give answer in not more than 100 words


Context:
{context}

Requested Section:
{query}

Return the exact section text below:
"""

        messages = [{"role": "user", "content": prompt}]

        # Shared semaphore: counts against the same 3-slot budget as ChatService
        async with _mistral_semaphore:
            response = await loop.run_in_executor(
                self.executor,
                lambda: self._call_mistral_with_retry(
                    model=settings.MISTRAL_MODEL,
                    messages=messages,
                )
            )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [doc["source"] for doc in results]
        }