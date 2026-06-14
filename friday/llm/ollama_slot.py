"""
friday/llm/ollama_slot.py
=========================
Slot 2: Local Ollama — the always-available fallback.

Wraps the existing ollama.AsyncClient().chat() call that all subsystems
previously called directly. Behaviour is identical to before.
"""

import asyncio
import logging
from typing import Optional

from friday.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaSlot(LLMProvider):
    """
    Ollama-backed LLM slot.

    Uses ollama.AsyncClient().chat() internally — same as the old direct calls.
    Guaranteed to be available when Ollama is running locally.
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    @property
    def name(self) -> str:
        return f"{self.model} (Ollama)"

    @property
    def is_available(self) -> bool:
        return True  # Ollama is always the last-resort fallback

    async def generate(
        self,
        messages: list,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> str:
        """
        Call Ollama. Maps directly to the old ollama.AsyncClient().chat() calls.
        """
        try:
            import ollama as _ollama
        except ImportError:
            logger.error("ollama package not installed.")
            return "Error: Ollama client not available."

        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            client = _ollama.AsyncClient()
            response = await asyncio.wait_for(
                client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                ),
                timeout=timeout,
            )
            return response["message"]["content"]
        except asyncio.TimeoutError:
            logger.debug(f"[OllamaSlot] Timed out after {timeout}s (model={self.model})")
            return f"I'm sorry Sir, that took too long. Please try again or use a simpler phrasing."
        except Exception as e:
            logger.error(f"[OllamaSlot] Call failed: {e}")
            return f"Error: {e}"

    async def generate_raw(self, prompt: str, *, timeout: float = 60.0) -> str:
        """
        Convenience: send a raw prompt string (wraps in user message).
        Used by legacy LLMClient.generate() compatibility shim.
        """
        try:
            import ollama as _ollama
        except ImportError:
            return "Error: Ollama client not available."

        try:
            client = _ollama.AsyncClient()
            response = await asyncio.wait_for(
                client.generate(model=self.model, prompt=prompt),
                timeout=timeout,
            )
            return response["response"]
        except asyncio.TimeoutError:
            return "Error: Ollama generate timed out."
        except Exception as e:
            logger.error(f"[OllamaSlot.generate_raw] {e}")
            return f"Error: {e}"
