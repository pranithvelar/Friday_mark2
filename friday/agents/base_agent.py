# BaseAgent — abstract base class for all agents
# NEW FILE
class BaseAgent:
    agent_id: str = ''
    description: str = ''
    capabilities: list = []
    tool_scope: list = []
    async def run(self, task: str, context: dict) -> dict:
        raise NotImplementedError
