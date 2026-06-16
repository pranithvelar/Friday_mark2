"""
LiveContextState — Continuous Background Context Engine
========================================================
The MAIN BRAIN must ALWAYS be aware of everything happening in the entire
program — whether or not the user is typing. This module provides that.

Instead of only building context when the user sends a message, this runs
a background asyncio loop that continuously refreshes a shared LiveContextState
object. Every LLM call reads from this object and injects it into the system
prompt — so the brain always has up-to-date awareness of:

  - Current active events / calendar
  - Any running execution (step progress, tool in use)
  - Recent memory updates
  - Conflict warnings
  - Pending plan approvals
  - Current system time

The background loop runs every REFRESH_INTERVAL_SECONDS.
The data is injected into AgentLoop._build_system_prompt() on EVERY LLM call.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │  LiveContextLoop (asyncio background task)       │
  │   every N seconds:                               │
  │     refresh facts / events / execution state     │
  │     → updates LiveContextState (thread-safe)     │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  AgentLoop._build_system_prompt()                │
  │   reads LiveContextState.as_system_block()       │
  │   → injects into EVERY LLM call, always         │
  └─────────────────────────────────────────────────┘
"""

import asyncio
import logging
import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 8   # Refresh live context every 8 seconds


class LiveContextState:
    """
    Thread-safe shared state container. Written by background loop,
    read by every LLM call. All fields are plain strings — cheap to read.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # ── Live fields ────────────────────────────────────────────────
        self.current_time_str:    str = ""
        self.itinerary_block:     str = ""   # upcoming events
        self.conflict_block:      str = ""   # contested events
        self.reminder_block:      str = ""   # day-before reminders
        self.execution_block:     str = ""   # running plan status
        self.pending_plan_block:  str = ""   # plan awaiting approval
        self.last_refreshed:      Optional[datetime.datetime] = None

    async def update(
        self,
        *,
        current_time_str: str = "",
        itinerary_block: str = "",
        conflict_block: str = "",
        reminder_block: str = "",
        execution_block: str = "",
        pending_plan_block: str = "",
    ):
        """Atomically update all live fields."""
        async with self._lock:
            self.current_time_str   = current_time_str
            self.itinerary_block    = itinerary_block
            self.conflict_block     = conflict_block
            self.reminder_block     = reminder_block
            self.execution_block    = execution_block
            self.pending_plan_block = pending_plan_block
            self.last_refreshed     = datetime.datetime.now()

    def as_system_block(self) -> str:
        """
        Returns a compact string block to inject into EVERY LLM system prompt.
        Only includes non-empty fields to keep token budget lean.
        This is read on every LLM call — must be fast (pure string concat).
        """
        parts = []

        if self.current_time_str:
            parts.append(f"[LIVE — Current Time: {self.current_time_str}]")

        if self.pending_plan_block:
            parts.append(self.pending_plan_block)

        if self.execution_block:
            parts.append(self.execution_block)

        if self.reminder_block:
            parts.append(self.reminder_block)

        if self.itinerary_block:
            parts.append(self.itinerary_block)

        if self.conflict_block:
            # Softened: LLM mentions once if relevant, doesn't hijack response
            parts.append(
                self.conflict_block.replace(
                    "[SYSTEM WARNING: CONFLICTING EVENTS DETECTED",
                    "[Note: scheduling conflict detected — mention once if relevant"
                )
            )

        return "\n".join(parts)


class LiveContextLoop:
    """
    Background asyncio task that continuously refreshes LiveContextState.

    Started once at application startup. Runs independently of user input.
    The brain is ALWAYS aware — not just when someone types.

    Usage:
        live_ctx = LiveContextState()
        loop_runner = LiveContextLoop(live_ctx, fact_store, state_manager)
        asyncio.create_task(loop_runner.run())
        # Then pass live_ctx to AgentLoop so it injects into system prompt
    """

    def __init__(
        self,
        live_ctx: LiveContextState,
        fact_store=None,
        state_manager=None,
        interval: int = REFRESH_INTERVAL_SECONDS,
    ):
        self.live_ctx      = live_ctx
        self.fact_store    = fact_store
        self.state_manager = state_manager
        self.interval      = interval
        self._running      = False

    def stop(self):
        self._running = False

    async def run(self):
        """
        Main background loop. Runs forever until stop() is called.
        Any single refresh failure is logged and silently swallowed —
        the brain loses one refresh cycle but never crashes.
        """
        self._running = True
        logger.info("[LiveContextLoop] Starting continuous context refresh loop")

        # First refresh immediately so the brain has context from the start
        await self._refresh_once()

        while self._running:
            await asyncio.sleep(self.interval)
            try:
                await self._refresh_once()
            except Exception as e:
                logger.warning(f"[LiveContextLoop] Refresh failed (non-fatal): {e}")

    async def _refresh_once(self):
        """Build the latest context snapshot and push it to LiveContextState."""
        now_dt = datetime.datetime.now()

        # ── 1. Current time (always available) ────────────────────────────────
        local_now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        current_time_str = local_now.strftime("%A, %B %d, %Y — %I:%M %p %Z").strip()

        # ── 2. Calendar / events / reminders ──────────────────────────────────
        itinerary_block = ""
        conflict_block  = ""
        reminder_block  = ""

        if self.fact_store:
            try:
                # Lint conflicts (fast SQL self-join)
                self.fact_store.lint_memory_conflicts()

                # Active events → itinerary block
                active_facts = self.fact_store.get_active_facts()
                if active_facts:
                    lines = []
                    for f in active_facts:
                        try:
                            ds = datetime.datetime.fromisoformat(
                                f["date_start"]
                            ).replace(tzinfo=None)
                            de = datetime.datetime.fromisoformat(
                                f["date_end"]
                            ).replace(tzinfo=None)
                            days = (ds.date() - now_dt.date()).days
                            if days < 0:
                                label = f"ONGOING (started {-days} days ago)"
                            elif days == 0:
                                label = f"TODAY {ds.strftime('%I:%M %p')}-{de.strftime('%I:%M %p')}"
                            elif days == 1:
                                label = f"TOMORROW {ds.strftime('%I:%M %p')}"
                            else:
                                label = f"{ds.strftime('%a %b %d')} ({days} days away)"
                            lines.append(f"  - {label}: {f['content']}")
                        except Exception:
                            pass
                    if lines:
                        itinerary_block = (
                            "[ABSOLUTE CONTINUOUS ITINERARY — ALL UPCOMING EVENTS]\n"
                            + "\n".join(lines)
                        )

                # Contested events → conflict block
                contested = self.fact_store.get_contested_facts()
                if contested:
                    lines = []
                    for f in contested:
                        try:
                            ds = datetime.datetime.fromisoformat(
                                f["date_start"]
                            ).replace(tzinfo=None)
                            days = (ds.date() - now_dt.date()).days
                            if days < 0:
                                continue
                            label = (
                                "TODAY" if days == 0
                                else "TOMORROW" if days == 1
                                else f"{days} days away"
                            )
                            lines.append(
                                f"  - {ds.strftime('%a %b %d')} ({label}): {f['content']}"
                            )
                        except Exception:
                            pass
                    if lines:
                        conflict_block = (
                            "[SYSTEM WARNING: CONFLICTING EVENTS DETECTED"
                            " — these events overlap and must be resolved]\n"
                            + "\n".join(lines)
                        )

                # Day-before reminders
                reminder_events = self.fact_store.get_events_needing_reminder()
                if reminder_events:
                    lines = []
                    for rev in reminder_events:
                        try:
                            ds = datetime.datetime.fromisoformat(
                                rev["date_start"]
                            ).replace(tzinfo=None)
                            hours = max(0, int((ds - now_dt).total_seconds() // 3600))
                            lines.append(
                                f"  - {ds.strftime('%A %b %d at %I:%M %p')} "
                                f"(in ~{hours}h): {rev['content']}"
                            )
                            self.fact_store.mark_reminder_sent(rev["id"])
                        except Exception:
                            pass
                    if lines:
                        reminder_block = (
                            "[\U0001f514 REMINDER — UPCOMING]\n"
                            "Proactively mention these to the user.\n"
                            + "\n".join(lines)
                        )
            except Exception as e:
                logger.debug(f"[LiveContextLoop] Calendar refresh error: {e}")

        # ── 3. Execution engine status ─────────────────────────────────────────
        execution_block = ""
        if self.state_manager:
            try:
                # Check all active executions
                for state in self.state_manager.active_executions.values():
                    block = state.get_context_for_llm()
                    if block:
                        execution_block = block
                        break   # Only show one execution at a time
            except Exception as e:
                logger.debug(f"[LiveContextLoop] Execution status error: {e}")

        # ── 4. Push to shared state (atomic) ───────────────────────────────────
        await self.live_ctx.update(
            current_time_str   = current_time_str,
            itinerary_block    = itinerary_block,
            conflict_block     = conflict_block,
            reminder_block     = reminder_block,
            execution_block    = execution_block,
            # pending_plan_block is set externally by MultiAgentPlanner
        )

        logger.debug(
            f"[LiveContextLoop] Refreshed at {current_time_str} "
            f"(events={bool(itinerary_block)}, "
            f"execution={bool(execution_block)}, "
            f"reminders={bool(reminder_block)})"
        )
