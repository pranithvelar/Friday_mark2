# BaseTool — abstract base class for all tools
# NEW FILE
class BaseTool:
    name: str = ''
    description: str = ''
    scope: str = 'general'
    parameters: dict = {}
    async def run(self, **kwargs) -> dict:
        raise NotImplementedError
