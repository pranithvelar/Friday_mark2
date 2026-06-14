"""
friday/agents/friday/llm.py
============================
Legacy LLMClient shim — kept for backward compatibility.

New code should use friday.llm.LLMProvider directly.
This class wraps a LLMProvider if one is injected, otherwise falls back
to the raw Ollama client (original behavior).
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Backward-compatible LLM client wrapper.

    When constructed with llm_provider, all calls are routed through it
    (enabling the API→Ollama slot system). Without it, uses Ollama directly.
    """

    def __init__(self, default_model: str = "llama3.1:8b", llm_provider=None):
        self.default_model = default_model
        self._provider = llm_provider

    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        if self._provider:
            return await self._provider.generate(messages)

        # Legacy direct ollama path
        try:
            import ollama
        except ImportError:
            return "Error: Ollama client not available."

        target_model = model or self.default_model
        try:
            client = ollama.AsyncClient()
            response = await client.chat(model=target_model, messages=messages)
            return response['message']['content']
        except Exception as e:
            logger.error(f"LLM Chat Error: {e}")
            return f"Thinking Error: {str(e)}"

    async def generate(self, prompt: str, model: Optional[str] = None) -> str:
        if self._provider:
            return await self._provider.generate([{"role": "user", "content": prompt}])

        try:
            import ollama
        except ImportError:
            return "Error: Ollama client not available."

        target_model = model or self.default_model
        try:
            client = ollama.AsyncClient()
            response = await client.generate(model=target_model, prompt=prompt)
            return response['response']
        except Exception as e:
            logger.error(f"LLM Generate Error: {e}")
            return f"Generation Error: {str(e)}"
