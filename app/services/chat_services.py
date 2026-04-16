import asyncio
import time
import random
from concurrent.futures import ThreadPoolExecutor

from app.utils.config import settings
from app.tools.legal_search_tool import LegalSearchTool

from mistralai import Mistral
from mistralai import models as mistral_models

# Global semaphore: only N requests call Mistral at the same time
_mistral_semaphore = asyncio.Semaphore(3)


class ChatService:
    def __init__(self):
        self.search_tool = LegalSearchTool()
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.executor = ThreadPoolExecutor()

    def _call_mistral_with_retry(self, model: str, messages: list, max_retries: int = 6) -> object:
        """
        Blocking Mistral call with exponential backoff on HTTP 429 rate-limit errors.
        Delays: ~2s, ~4s, ~8s, ~16s, ~32s, ~64s (+ small jitter each time).
        """
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

    async def get_answer(self, query: str):
        loop = asyncio.get_event_loop()

        # Run blocking search in thread
        results = await loop.run_in_executor(
            self.executor,
            self.search_tool.search,
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

        messages = [{"role": "user", "content": prompt}]

        # Throttle: wait here until a slot is free before calling Mistral
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