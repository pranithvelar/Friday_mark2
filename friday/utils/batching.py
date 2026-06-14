import asyncio
from typing import List, Any, Callable, Coroutine
import logging

logger = logging.getLogger(__name__)

class AsyncBatcher:
    """
    Performance optimizer to prevent local LLM (Ollama) overload.
    Collects rapid sequential embeddings or chat requests and processes them
    in constrained concurrency batches to avoid OOM or timeout errors.
    """
    def __init__(self, max_concurrency: int = 3):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_in_batches(self, items: List[Any], func: Callable[[Any], Coroutine]) -> List[Any]:
        """
        Runs `func` over `items`, limiting max concurrent executions.
        """
        async def _worker(item: Any):
            async with self.semaphore:
                try:
                    return await func(item)
                except Exception as e:
                    logger.error(f"Batch execution failed for item: {e}")
                    return None

        # Execute everything with bound concurrency
        tasks = [_worker(item) for item in items]
        return await asyncio.gather(*tasks)
