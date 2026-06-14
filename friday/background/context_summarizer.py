"""
BACKGROUND_WORKER/context_summarizer.py

Runs post-response compaction as a fire-and-forget background task.
This prevents the "[Summarizing older context...]" message from blocking
the main response path. The summary is pre-computed and ready in the DB
before the user sends their NEXT message.

Architecture:
  - After each agent response, loop.py calls trigger_background_compact()
  - This creates an asyncio.Task (non-blocking) that runs _maybe_compact()
  - The main response returns immediately with no delay
  - On the user's next turn, the summary is already cached in meta table

Future-proof:
  - TTS/STT: no changes needed — output path is unchanged
  - Gemini API: pass `client` param to use any LLM, not just Ollama
  - MCPs/n8n: compaction is isolated, no coupling to external integrations
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Global flag so only one compaction runs at a time even if messages come fast
_compaction_running = False


async def _run_compact_task(agent_loop) -> None:
    """Internal coroutine that actually runs compaction."""
    global _compaction_running
    try:
        _compaction_running = True
        await agent_loop._maybe_compact()
        logger.debug("[BackgroundCompactor] Compaction completed.")
    except Exception as e:
        logger.warning(f"[BackgroundCompactor] Compaction failed silently: {e}")
    finally:
        _compaction_running = False


def trigger_background_compact(agent_loop) -> None:
    """
    Call this AFTER the agent response is returned to the user.
    Creates a fire-and-forget asyncio task — zero blocking.
    Safe to call every turn; skips if already running.
    """
    global _compaction_running
    if _compaction_running:
        logger.debug("[BackgroundCompactor] Skipping — compaction already in progress.")
        return

    # Schedule on the running event loop — non-blocking
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_run_compact_task(agent_loop))
    except RuntimeError:
        # No running event loop (e.g., called from sync context) — skip gracefully
        logger.debug("[BackgroundCompactor] No running event loop, skipping.")
