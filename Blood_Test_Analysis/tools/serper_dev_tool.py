from crewai.tools import BaseTool

class SerperDevTool(BaseTool):
    name: str = "search_tool"
    description: str = "Searches the web for a query and returns results."

    def _run(self, query: str) -> str:
        # Dummy result until you implement real search
        return f"Fake search result for: {query}"

if __name__ == "__main__":
    tool = SerperDevTool()
    result = tool._run("What is anemia?")
    print(result)
