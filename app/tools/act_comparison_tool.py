from app.services.comparison_service import ComparisonService

class ActComparisonTool:
    def __init__(self):
        self.comparison_service = ComparisonService()

    async def compare(self, topic: str, act_a: str, act_b: str):
        return await self.comparison_service.compare_acts(topic, act_a, act_b)