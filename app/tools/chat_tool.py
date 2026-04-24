from app.services.chat_services import ChatService

class ChatTool:
    def __init__(self):
        self.chat_service = ChatService()

    async def ask(self, query: str) -> dict:
        return await self.chat_service.get_answer(query)