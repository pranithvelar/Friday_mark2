"""
friday/llm/api_slot.py
======================
Slot 1: Any OpenAI-compatible API endpoint.

Why OpenAI-compat?
  The OpenAI Chat Completions protocol (POST /v1/chat/completions) has
  become the industry standard. Every major provider supports it:

  Provider       base_url
  ─────────────────────────────────────────────────────────────────
  OpenAI         (leave blank — openai package default)
  OpenRouter     https://openrouter.ai/api/v1
  Gemini         https://generativelanguage.googleapis.com/v1beta/openai/
  DeepSeek       https://api.deepseek.com/v1
  Groq           https://api.groq.com/openai/v1
  Mistral        https://api.mistral.ai/v1
  Together AI    https://api.together.xyz/v1
  Perplexity     https://api.perplexity.ai
  Claude*        Use OpenRouter (openrouter.ai routes to all Anthropic models)

  *Direct Anthropic API uses a different protocol. Recommend OpenRouter
   for Claude access to keep a single integration.

Configuration (via .env or environment variables):
  LLM_API_KEY=sk-or-...       # your key (any provider)
  LLM_API_BASE_URL=...        # provider base URL (blank = OpenAI)
  LLM_API_MODEL=gpt-4o-mini   # model name (provider-specific)
"""

import asyncio
import logging
from typing import Optional

from friday.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class APISlot(LLMProvider):
    """
    OpenAI-compatible API slot.

    Supports any provider that speaks the OpenAI Chat Completions protocol:
    OpenAI, OpenRouter, Gemini, DeepSeek, Groq, Mistral, Together, etc.

    is_available is a CHEAP LOCAL CHECK — it returns False if the API key
    is blank, with no network call.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            model:    Model name (provider-specific, e.g. "gpt-4o-mini",
                      "anthropic/claude-3-5-sonnet", "deepseek-chat", etc.)
            api_key:  API key for the provider.
            base_url: Provider base URL. None = OpenAI native endpoint.
        """
        self.model = model
        self._api_key = api_key.strip() if api_key else ""
        self._base_url = base_url.strip() if base_url else None

    @property
    def name(self) -> str:
        base = self._base_url or "openai"
        return f"{self.model} (API:{base})"

    @property
    def is_available(self) -> bool:
        """Cheap local check — no network call."""
        return bool(self._api_key)

    async def generate(
        self,
        messages: list,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> str:
        """
        Call the OpenAI-compatible endpoint.

        Raises exceptions on failure so SlottedProvider can catch them
        and trigger the Ollama fallback.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,  # None → uses openai default
        )

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=timeout,
        )
        content = response.choices[0].message.content
        return content or ""
