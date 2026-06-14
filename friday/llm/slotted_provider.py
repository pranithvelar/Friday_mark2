"""
friday/llm/slotted_provider.py
===============================
The brain of the fallback system.

Tries Slot 1 (API). Falls back to Slot 2 (Ollama) on any failure.
This is the ONLY object that callers (AgentLoop, LLMRouter, etc.) hold.

Fallback triggers:
  - Slot 1 api_key is empty (skipped entirely, zero latency)
  - openai.AuthenticationError  (401 — bad key)
  - openai.PermissionDeniedError (403 — insufficient permissions)
  - openai.RateLimitError        (429 — quota exceeded)
  - openai.APIConnectionError    (network error)
  - asyncio.TimeoutError         (slot 1 too slow)
  - Any other exception          (catch-all safety net)

The callers (agent, router, dreamer, etc.) see identical behaviour
regardless of which slot is actually serving the request.
"""

import asyncio
import logging
from typing import Optional

from friday.llm.base import LLMProvider
from friday.llm.api_slot import APISlot
from friday.llm.ollama_slot import OllamaSlot

logger = logging.getLogger(__name__)


class SlottedProvider(LLMProvider):
    """
    Two-slot LLM provider with automatic fallback.

    slot1: APISlot (optional) — any OpenAI-compatible cloud provider
    slot2: OllamaSlot         — local Ollama, always available

    Public interface is identical to LLMProvider.generate().
    Callers never need to know which slot handled the request.
    """

    def __init__(
        self,
        slot1: Optional[APISlot],
        slot2: OllamaSlot,
    ):
        self._slot1 = slot1
        self._slot2 = slot2
        # Tracks which slot handled the last successful request
        self._last_used: str = ""

    @property
    def name(self) -> str:
        return self.active_slot_name

    @property
    def active_slot_name(self) -> str:
        """
        Returns the name of the slot that would be tried first right now.
        If slot1 is available → slot1.name, else → slot2.name.
        """
        if self._slot1 and self._slot1.is_available:
            return self._slot1.name
        return self._slot2.name

    @property
    def is_available(self) -> bool:
        return True  # Ollama always available as last resort

    async def generate(
        self,
        messages: list,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> str:
        """
        Try Slot 1 (API). Fall back to Slot 2 (Ollama) on any failure.

        This method NEVER raises. It always returns a string.
        """
        # ── Slot 1: API ──────────────────────────────────────────────────────
        if self._slot1 and self._slot1.is_available:
            try:
                result = await self._slot1.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                self._last_used = self._slot1.name
                logger.debug(f"[SlottedProvider] Slot 1 served: {self._slot1.name}")
                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"[SlottedProvider] API slot timed out ({timeout}s) — falling back to Ollama"
                )
            except Exception as e:
                # Catch everything: auth errors, quota errors, network errors, etc.
                logger.warning(
                    f"[SlottedProvider] API slot failed ({type(e).__name__}: {e}) "
                    f"— falling back to Ollama"
                )

        # ── Slot 2: Ollama fallback ───────────────────────────────────────────
        logger.debug(f"[SlottedProvider] Slot 2 serving: {self._slot2.name}")
        result = await self._slot2.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._last_used = self._slot2.name
        return result

    def last_used(self) -> str:
        """Which slot served the most recent request."""
        return self._last_used

    def get_status(self) -> dict:
        """Diagnostic summary — useful for `status` command in terminal."""
        return {
            "slot1": self._slot1.name if self._slot1 else "disabled",
            "slot1_available": self._slot1.is_available if self._slot1 else False,
            "slot2": self._slot2.name,
            "last_used": self._last_used or "none yet",
        }
