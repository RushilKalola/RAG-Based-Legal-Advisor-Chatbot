from app.tools.legal_search_tool import LegalSearchTool


class SectionLookupTool:
    def __init__(self):
        self.search_tool = LegalSearchTool()

    def lookup(self, section_query: str):
        """
        Example:
        'IPC Section 420'
        'Article 21 Constitution'
        """

        results = self.search_tool.search(section_query)

        if not results:
            return {"text": "No relevant section found.", "source": ""}

        # Return best match
        best = results[0]

        return {
            "text": best["text"],
            "source": best["source"]
        }