# BaseInterface — abstract input/output channel
# NEW FILE
class BaseInterface:
    async def receive_input(self) -> str:
        raise NotImplementedError
    async def send_output(self, text: str):
        raise NotImplementedError
