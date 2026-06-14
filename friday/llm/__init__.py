"""
friday/llm/__init__.py
======================
Public API for the slot-based LLM provider.

Usage (from chat.py startup):
    from friday.llm import build_provider
    provider = build_provider(config)   # SlottedProvider
    print(provider.active_slot_name)    # "gemini-2.5-flash (API)" or "llama3.1:8b (Ollama)"
"""

from friday.llm.base import LLMProvider
from friday.llm.ollama_slot import OllamaSlot
from friday.llm.api_slot import APISlot
from friday.llm.slotted_provider import SlottedProvider


def build_provider(config) -> "SlottedProvider":
    """
    Build a SlottedProvider from IntelligentMemoryConfig.

    Slot 1 (API): active when config.llm_api_key is non-empty.
      Reads base_url from config.llm_api_base_url.
      Default base_url = None → uses native OpenAI endpoint.
      OpenRouter: set base_url = https://openrouter.ai/api/v1
      Gemini:     set base_url = https://generativelanguage.googleapis.com/v1beta/openai/
      DeepSeek:   set base_url = https://api.deepseek.com/v1

    Slot 2 (Ollama): always available. Uses config.llama_model.

    Fallback logic (in SlottedProvider):
      → missing/empty api_key → skip slot 1, go straight to Ollama
      → slot 1 call fails (401/429/quota/network/timeout) → fall back to Ollama
    """
    ollama_slot = OllamaSlot(model=config.llama_model)

    api_key = getattr(config, "llm_api_key", "").strip()
    api_model = getattr(config, "llm_api_model", "gpt-4o-mini")
    api_base_url = getattr(config, "llm_api_base_url", "") or None  # empty str → None

    if api_key:
        api_slot = APISlot(
            model=api_model,
            api_key=api_key,
            base_url=api_base_url,
        )
    else:
        api_slot = None

    return SlottedProvider(slot1=api_slot, slot2=ollama_slot)


__all__ = ["LLMProvider", "OllamaSlot", "APISlot", "SlottedProvider", "build_provider"]
