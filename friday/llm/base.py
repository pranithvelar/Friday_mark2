"""
friday/llm/base.py
==================
Abstract base class for all LLM providers.

All slots (OllamaSlot, APISlot) implement this interface.
SlottedProvider also implements it so callers treat it identically.

The contract:
  - generate(messages, ...) -> str  (always a plain string, never raises)
  - name property (human-readable slot description)
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """
    Unified LLM interface. One method: generate().

    Input:  OpenAI-style messages list + optional params.
    Output: Plain string (assistant response content).

    Implementors MUST NOT raise — they should return an error string
    or empty string if they fail, so callers never crash.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> str:
        """
        Send a chat completion request. Return the response text.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature. Use 0.0 for deterministic outputs.
            max_tokens:  Max tokens in response. None = provider default.
            timeout:     Hard timeout in seconds.

        Returns:
            Response text string. Never raises.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier: e.g. 'gemini-2.5-flash (API)'"""
        ...

    @property
    def is_available(self) -> bool:
        """Whether this slot can be used. Override to add health checks."""
        return True
