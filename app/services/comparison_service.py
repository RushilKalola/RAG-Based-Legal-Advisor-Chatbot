import asyncio

from app.utils.config import settings
from app.services.retrieval import RetrievalService
from app.services.chat_services import _mistral_semaphore

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage


class ComparisonService:
    def __init__(self):
        self.retrieval = RetrievalService()
        self.client = ChatMistralAI(
            api_key=settings.MISTRAL_API_KEY,
            model=settings.MISTRAL_MODEL,
            max_retries=6,
        )

    def _search_for_act(self, topic: str, act_name: str) -> list:
        """Search Qdrant for chunks from a specific act on a topic."""
        query = f"{topic} {act_name}"
        results = self.retrieval.search(query)

        # Filter results to only chunks whose source mentions the act name
        act_keywords = act_name.lower().split()
        filtered = [
            r for r in results
            if any(kw in r["source"].lower() for kw in act_keywords)
        ]

        # If the filter is too aggressive, fall back to all results
        return filtered if filtered else results

    async def compare_acts(self, topic: str, act_a: str, act_b: str) -> dict:

        # Run both searches concurrently
        results_a, results_b = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(None, self._search_for_act, topic, act_a),
            asyncio.get_event_loop().run_in_executor(None, self._search_for_act, topic, act_b),
        )

        if not results_a and not results_b:
            return {
                "topic": topic,
                "act_a": act_a,
                "act_b": act_b,
                "comparison": "No relevant content found for either act.",
                "sources_a": [],
                "sources_b": [],
            }

        context_a = "\n\n".join([doc["text"] for doc in results_a]) or "No content found."
        context_b = "\n\n".join([doc["text"] for doc in results_b]) or "No content found."

        prompt = f"""You are a legal comparison assistant specializing in Indian law.

Compare how the two acts below treat the topic: "{topic}"

---
ACT A — {act_a}:
{context_a}

---
ACT B — {act_b}:
{context_b}

---
Instructions:
- Produce a structured comparison with these four sections:
  1. **{act_a} — Key Provisions**: Quote or closely paraphrase the most relevant text from Act A.
  2. **{act_b} — Key Provisions**: Quote or closely paraphrase the most relevant text from Act B.
  3. **Similarities**: What do both acts agree on regarding "{topic}"?
  4. **Differences**: Where do they diverge — in scope, penalty, procedure, or definitions?
- Be specific and cite section numbers when visible in the context.
- If no relevant content was found for one act, say so clearly in that section.
- Do not invent provisions that are not in the provided context."""

        messages = [HumanMessage(content=prompt)]

        async with _mistral_semaphore:
            response = await self.client.ainvoke(messages)

        return {
            "topic": topic,
            "act_a": act_a,
            "act_b": act_b,
            "comparison": response.content,
            "sources_a": list({doc["source"] for doc in results_a}),
            "sources_b": list({doc["source"] for doc in results_b}),
        }
    