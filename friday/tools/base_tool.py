# BaseTool — abstract base class for all tools in friday/tools/
# Subclass this to add a new tool. ToolRegistry auto-discovers subclasses.
#
# Usage:
#   class WebSearchTool(BaseTool):
#       name        = "web_search"
#       description = "Search the web using DuckDuckGo."
#       scope       = "general"   # "general" = medium+complex, "brain" = AgentLoop only
#       parameters  = {
#           "type": "object",
#           "properties": {
#               "query": {"type": "string", "description": "Search query"}
#           },
#           "required": ["query"]
#       }
#       async def run(self, **kwargs) -> dict:
#           query = kwargs.get("query", "")
#           # ... real implementation ...
#           return {"result": "..."}

class BaseTool:
    name:        str  = ""          # Unique tool name. Must match what LLM will call.
    description: str  = ""          # One-line description shown to the LLM.
    scope:       str  = "general"   # "general" | "brain" — controls which tiers see it.
    parameters:  dict = {}          # JSON Schema for parameters (type/properties/required).

    async def run(self, **kwargs) -> dict:
        """Execute the tool. Override this in every subclass."""
        raise NotImplementedError(f"Tool '{self.name}' has not implemented run()")

    def to_schema(self) -> dict:
        """
        Return the JSON schema dict for this tool as expected by AgentLoop.
        Wraps self.parameters in the correct structure automatically.
        """
        # If caller already provided the full schema with 'type', use it directly.
        params = self.parameters or {}
        if "type" not in params:
            # Bare dict of properties was given — wrap it.
            params = {
                "type": "object",
                "properties": params,
                "required": []
            }
        return {
            "name":        self.name,
            "description": self.description,
            "parameters":  params,
        }
