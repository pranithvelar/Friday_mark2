"""
test_llm_slots.py
==================
Tests for the slot-based LLM provider system.
Run: python test_llm_slots.py
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []

def check(name, condition, detail=""):
    icon = PASS if condition else FAIL
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))


# ─────────────────────────────────────────────────────────────────
# 1. Config loads LLM slot fields
# ─────────────────────────────────────────────────────────────────
def test_config():
    print("\n[1] Config & .env loading")
    from friday.config.settings import IntelligentMemoryConfig
    cfg = IntelligentMemoryConfig.load()
    check("Config loads without error", True)
    check("llm_api_key field exists", hasattr(cfg, "llm_api_key"))
    check("llm_api_base_url field exists", hasattr(cfg, "llm_api_base_url"))
    check("llm_api_model field exists", hasattr(cfg, "llm_api_model"))
    check("llama_model field exists", hasattr(cfg, "llama_model"), cfg.llama_model)
    print(f"  {INFO} API key: {'SET (' + cfg.llm_api_model + ')' if cfg.llm_api_key else 'NOT SET — Ollama only'}")
    print(f"  {INFO} Ollama model: {cfg.llama_model}")
    return cfg


# ─────────────────────────────────────────────────────────────────
# 2. OllamaSlot
# ─────────────────────────────────────────────────────────────────
async def test_ollama_slot():
    print("\n[2] OllamaSlot")
    from friday.llm.ollama_slot import OllamaSlot
    slot = OllamaSlot(model="llama3.1:8b")
    check("OllamaSlot.is_available == True", slot.is_available)
    check("OllamaSlot.name contains 'Ollama'", "Ollama" in slot.name)

    print(f"  {INFO} Calling Ollama (llama3.1:8b)...")
    try:
        result = await slot.generate([{"role": "user", "content": "Reply with exactly: PONG"}],
                                     temperature=0.0, timeout=30)
        check("OllamaSlot returns non-empty string", bool(result and result.strip()),
              repr(result[:80]))
    except Exception as e:
        check("OllamaSlot live call", False, str(e))


# ─────────────────────────────────────────────────────────────────
# 3. APISlot availability (no API key = unavailable)
# ─────────────────────────────────────────────────────────────────
def test_api_slot_availability():
    print("\n[3] APISlot availability checks")
    from friday.llm.api_slot import APISlot

    slot_empty = APISlot(model="gpt-4o-mini", api_key="", base_url=None)
    check("APISlot with empty key → is_available=False", not slot_empty.is_available)

    slot_spaces = APISlot(model="gpt-4o-mini", api_key="   ", base_url=None)
    check("APISlot with whitespace key → is_available=False", not slot_spaces.is_available)

    slot_valid = APISlot(model="gpt-4o-mini", api_key="sk-fake-key", base_url=None)
    check("APISlot with non-empty key → is_available=True", slot_valid.is_available)


# ─────────────────────────────────────────────────────────────────
# 4. SlottedProvider fallback logic (mock Slot 1 to fail)
# ─────────────────────────────────────────────────────────────────
async def test_slotted_fallback():
    print("\n[4] SlottedProvider fallback logic")
    from friday.llm.ollama_slot import OllamaSlot
    from friday.llm.api_slot import APISlot
    from friday.llm.slotted_provider import SlottedProvider

    # Slot 1 with a deliberately bad API key (will fail on call)
    bad_slot1 = APISlot(model="gpt-4o-mini", api_key="sk-intentionally-invalid", base_url=None)
    ollama_slot = OllamaSlot(model="llama3.1:8b")
    provider = SlottedProvider(slot1=bad_slot1, slot2=ollama_slot)

    check("SlottedProvider.is_available == True (Ollama always available)", provider.is_available)
    check("active_slot_name reflects slot1 when key set", "gpt-4o-mini" in provider.active_slot_name)

    print(f"  {INFO} Calling with bad API key (expect fallback to Ollama)...")
    try:
        result = await provider.generate(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            temperature=0.0,
            timeout=30,
        )
        check("SlottedProvider falls back and returns response", bool(result and result.strip()),
              repr(result[:80]))
        check("Last used slot is Ollama", "Ollama" in provider.last_used())
    except Exception as e:
        check("SlottedProvider fallback", False, str(e))


# ─────────────────────────────────────────────────────────────────
# 5. build_provider from config
# ─────────────────────────────────────────────────────────────────
def test_build_provider(cfg):
    print("\n[5] build_provider factory")
    from friday.llm import build_provider
    provider = build_provider(cfg)
    check("build_provider returns SlottedProvider", provider is not None)
    print(f"  {INFO} Active slot: {provider.active_slot_name}")
    check("active_slot_name is non-empty string", bool(provider.active_slot_name))
    status = provider.get_status()
    check("get_status() returns dict", isinstance(status, dict))
    print(f"  {INFO} Status: {status}")
    return provider


# ─────────────────────────────────────────────────────────────────
# 6. Live API slot test (only if key is set)
# ─────────────────────────────────────────────────────────────────
async def test_api_slot_live(cfg):
    print("\n[6] Live API slot test")
    if not cfg.llm_api_key:
        print(f"  {INFO} LLM_API_KEY not set — skipping live API test.")
        print(f"  {INFO} To test: set LLM_API_KEY in .env and re-run.")
        return

    from friday.llm.api_slot import APISlot
    slot = APISlot(
        model=cfg.llm_api_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_api_base_url or None,
    )
    check("APISlot.is_available", slot.is_available)
    print(f"  {INFO} Calling {slot.name}...")
    try:
        result = await slot.generate(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            temperature=0.0,
            timeout=30,
        )
        check("APISlot live call returns non-empty", bool(result and result.strip()),
              repr(result[:80]))
    except Exception as e:
        check("APISlot live call", False, str(e))


# ─────────────────────────────────────────────────────────────────
# 7. End-to-end: AgentLoop with slotted provider
# ─────────────────────────────────────────────────────────────────
async def test_agent_loop_e2e(provider):
    print("\n[7] AgentLoop end-to-end with slotted provider")
    from friday.agents.friday.agent import AgentLoop
    loop = AgentLoop(
        workspace_dir="",
        model="llama3.1:8b",
        llm_provider=provider,
    )
    check("AgentLoop accepts llm_provider", loop._llm_provider is provider)

    print(f"  {INFO} Running a single LLM call through AgentLoop._llm_call()...")
    try:
        result = await loop._llm_call([{"role": "user", "content": "Say: HELLO"}])
        check("AgentLoop._llm_call returns non-empty string", bool(result and result.strip()),
              repr(result[:80]))
    except Exception as e:
        check("AgentLoop._llm_call", False, str(e))


# ─────────────────────────────────────────────────────────────────
# 8. LLMRouter accepts llm_provider
# ─────────────────────────────────────────────────────────────────
async def test_llm_router(provider):
    print("\n[8] LLMRouter with slotted provider")
    from friday.router.llm_router import LLMRouter
    from friday.router.intent_classifier import FastIntentClassifier
    router = LLMRouter(
        model="llama3.1:8b",
        fallback_classifier=FastIntentClassifier(),
        llm_provider=provider,
    )
    check("LLMRouter accepts llm_provider", router._llm_provider is provider)

    print(f"  {INFO} Classifying a query...")
    try:
        complexity, category = await router.classify("what time is it?", session_id="test")
        check("LLMRouter.classify returns result", complexity is not None,
              f"{complexity.value} / {category.value}")
    except Exception as e:
        check("LLMRouter.classify", False, str(e))


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  FRIDAY LLM SLOT SYSTEM — TEST SUITE")
    print("=" * 60)

    cfg = test_config()
    await test_ollama_slot()
    test_api_slot_availability()
    await test_slotted_fallback()
    provider = test_build_provider(cfg)
    await test_api_slot_live(cfg)
    await test_agent_loop_e2e(provider)
    await test_llm_router(provider)

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed")
    if passed == total:
        print("  \033[92mAll tests passed!\033[0m")
    else:
        failed = [n for n, ok in results if not ok]
        print(f"  \033[91mFailed: {failed}\033[0m")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
