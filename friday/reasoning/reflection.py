"""
Reflection Agent — Self-Critique Loop
======================================
Friday's internal diagnostic system. This is NOT the same as _maybe_reflect() in agent.py.

  _maybe_reflect()  → learns about the USER (facts, preferences) every 12 msgs → Layer 6
  ReflectionAgent   → learns about FRIDAY'S OWN FAILURES (tool errors, user corrections) → Layer 5

When Friday fails to execute a tool correctly, or when the user explicitly corrects her,
this module:
  1. Receives the failure context (recent messages + what failed)
  2. Fires ONE focused LLM diagnostic call (temp=0.0 — deterministic, not creative)
  3. Extracts a structured lesson: {trigger, fix, confidence}
  4. Writes it to Layer 5 (ProceduralMemory) as a behavioral pattern

The lesson auto-injects into the NEXT relevant LLM context via ContextAssembler,
which already reads ProceduralMemory on every request. No new injection wiring needed.

Design principles (Tony Stark / JARVIS style):
  - Fire-and-forget : always run as asyncio.create_task() — zero latency impact on user
  - Non-fatal       : every exception is caught and logged — never disrupts the user
  - Cheap gate      : check_for_correction() is a regex check (0ms) before any LLM call
  - Idempotent      : ProceduralMemory.add_pattern() deduplicates — safe to call repeatedly
  - Deterministic   : temperature=0.0 on the diagnostic call — we want the most likely lesson

Scalability note:
  To add a new failure signal in the future (e.g., plan step timeout in MultiAgentPlanner):
    1. Add an `on_plan_timeout(...)` method here — follows the same _diagnose_and_learn pattern
    2. Add one asyncio.create_task(...) call in the new signal location
    3. Optionally inject _reflection_agent into that component via chat.py
  Zero changes to any other existing system.
"""

import re
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ── Correction detection: regex gate (runs BEFORE any LLM call) ─────────────
# These patterns mean the user is explicitly correcting Friday's behavior.
# Tuned for LOW false-positive rate — better to miss a correction than to
# fire an LLM call on "call the wrong number" or "wrong turn".
CORRECTION_PATTERNS = re.compile(
    r'\b('
    r"you(?:'re| are) wrong"
    r"|that(?:'s| is) (?:not right|wrong|incorrect|not what i)"
    r"|not what i (?:said|meant|asked|told you)"
    r"|you (?:got it|used the) wrong"
    r"|wrong (?:date|time|format|syntax|tool|approach)"
    r"|that(?:'s| is) not (?:how|what|the right)"
    r"|no[,\s]+(?:i (?:said|meant|asked)|that(?:'s| is) not)"
    r"|you(?:'re| are) (?:using|doing) it wrong"
    r"|the (?:date|time|format|syntax) (?:is|was) wrong"
    r"|that was (?:wrong|incorrect|not right)"
    r')\b',
    re.IGNORECASE,
)


# ── Diagnostic prompt (filled at call time) ──────────────────────────────────
REFLECTION_SYSTEM = """You are Friday's internal self-diagnostic system.
A failure just occurred. Your job: extract ONE actionable lesson to prevent recurrence.

FAILURE TYPE: {failure_type}
FAILURE DETAIL: {failure_detail}

RECENT CONVERSATION (for context):
{conversation}

Analyze what went wrong and extract a lesson.

Output ONLY valid JSON, nothing else:
{{
  "lesson": "One sentence describing what went wrong",
  "trigger": "When [specific situation that caused the failure — under 10 words]",
  "fix": "Always [specific corrected behavior — under 15 words]",
  "confidence": 0.7
}}

Rules:
- trigger and fix must be SPECIFIC and ACTIONABLE, not generic platitudes
- Focus on tool-use, formatting, date/time handling, and behavioral corrections
- confidence: 0.5 for inferred tool failures | 0.7 for explicit user corrections | 0.85 for repeated failures
- If nothing actionable can be extracted: {{"lesson": "", "trigger": "", "fix": "", "confidence": 0}}
- Do NOT include any text outside the JSON object
"""

# Minimum confidence required to write a lesson to Layer 5
LESSON_MIN_CONFIDENCE = 0.5

# How many recent history messages to include in the diagnostic prompt
REFLECTION_CONTEXT_MESSAGES = 6


class ReflectionAgent:
    """
    Friday's self-critique engine. Converts failures into Layer 5 behavioral lessons.

    Injected into AgentLoop as `loop._reflection_agent` after construction in chat.py,
    following the exact same pattern as `loop._knowledge_extractor`.

    Public API:
        await on_tool_failure(history, failed_tool, fallback_reason)
            → called by AgentLoop when tool loop exits without a clean answer

        await check_for_correction(user_message, history)
            → called before every LLM call in AgentLoop (cheap regex gate)

        await on_user_correction(history, correction_text)
            → called internally by check_for_correction when keywords match

    Future extension pattern (zero changes to existing systems):
        1. Add on_<new_signal>(self, ...) method here
        2. Call asyncio.create_task(self._reflection_agent.on_<new_signal>(...)) at the signal site
        Done. Everything else is unchanged.
    """

    def __init__(self, llm_provider, procedural_memory):
        """
        Parameters
        ----------
        llm_provider      : LLMProvider — shared with AgentLoop (no new model instantiation)
        procedural_memory : ProceduralMemory — Layer 5 where lessons are written
        """
        self.llm = llm_provider
        self.l5  = procedural_memory
        # Concurrency guard — prevents two diagnostics running simultaneously
        # (harmless if one is missed — idempotent writes)
        self._running: bool = False

    # ── Public: Tool failure signal ──────────────────────────────────────────

    async def on_tool_failure(
        self,
        history: List[Dict[str, Any]],
        failed_tool: str,
        fallback_reason: str,
    ) -> None:
        """
        Called by AgentLoop when the tool loop exits without a clean answer.
        Diagnoses WHY the failure occurred and writes a lesson to Layer 5.

        Always returns None. Never raises. Fire-and-forget via create_task().
        """
        if self._running:
            logger.debug("[ReflectionAgent] Skipped on_tool_failure — previous diagnostic active")
            return

        failure_detail = (
            f"Tool attempted: '{failed_tool or 'unknown'}'. "
            f"Exit reason: {fallback_reason}."
        )
        await self._diagnose_and_learn(
            failure_type          = "tool_execution_failure",
            failure_detail        = failure_detail,
            history               = history,
            context_tag           = "tool_usage",
            base_confidence_boost = 0.0,
        )

    # ── Public: User correction gateway ─────────────────────────────────────

    async def check_for_correction(
        self,
        user_message: str,
        history: List[Dict[str, Any]],
    ) -> None:
        """
        Cheap gateway — called before every LLM call in AgentLoop.
        Cost model:
          - No correction keywords → returns instantly, zero LLM cost
          - Correction keywords matched → fires ONE diagnostic LLM call

        Always returns None. Never raises.
        """
        if not CORRECTION_PATTERNS.search(user_message):
            return   # Fast path: ~0ms, no LLM cost

        logger.debug(f"[ReflectionAgent] Correction keyword detected: '{user_message[:80]}'")
        await self.on_user_correction(
            history         = history,
            correction_text = user_message,
        )

    # ── Public: Explicit user correction signal ──────────────────────────────

    async def on_user_correction(
        self,
        history: List[Dict[str, Any]],
        correction_text: str,
    ) -> None:
        """
        Called when the user explicitly corrects Friday's behavior.
        Gets higher confidence than inferred tool failures because the user
        directly stated what was wrong.

        Always returns None. Never raises. Fire-and-forget via create_task().
        """
        if self._running:
            logger.debug("[ReflectionAgent] Skipped on_user_correction — previous diagnostic active")
            return

        await self._diagnose_and_learn(
            failure_type          = "user_correction",
            failure_detail        = f"User explicitly corrected Friday: \"{correction_text[:200]}\"",
            history               = history,
            context_tag           = "behavior",
            base_confidence_boost = 0.15,   # user stated it directly → higher confidence
        )

    # ── Internal: Core diagnostic pipeline ──────────────────────────────────

    async def _diagnose_and_learn(
        self,
        failure_type: str,
        failure_detail: str,
        history: List[Dict[str, Any]],
        context_tag: str = "tool_usage",
        base_confidence_boost: float = 0.0,
    ) -> None:
        """
        Core pipeline: build prompt → LLM call (temp=0.0) → parse → write to L5.
        All exceptions are caught — this method never raises.
        """
        if self._running:
            return

        self._running = True
        try:
            # 1. Format recent conversation (capped at REFLECTION_CONTEXT_MESSAGES)
            conversation = self._format_history(history, max_messages=REFLECTION_CONTEXT_MESSAGES)

            # 2. Build diagnostic prompt
            system_prompt = REFLECTION_SYSTEM.format(
                failure_type   = failure_type,
                failure_detail = failure_detail,
                conversation   = conversation,
            )

            # 3. LLM diagnostic call — temperature=0.0 for deterministic lesson extraction
            raw = await self.llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": "Analyze the failure and output the lesson JSON."},
                ],
                temperature=0.0,
            )

            if not raw or not raw.strip():
                logger.debug("[ReflectionAgent] LLM returned empty response — nothing to learn")
                return

            # 4. Parse lesson from LLM output
            lesson = self._parse_lesson(raw)
            if not lesson:
                logger.debug(f"[ReflectionAgent] Could not parse lesson JSON from: {raw[:120]}")
                return

            trigger    = lesson.get("trigger", "").strip()
            fix        = lesson.get("fix", "").strip()
            confidence = float(lesson.get("confidence", 0.5)) + base_confidence_boost
            confidence = min(0.95, confidence)   # cap: reflection isn't ground truth

            # 5. Validate — skip empty or below-threshold lessons
            if not trigger or not fix:
                logger.debug("[ReflectionAgent] Lesson trigger or fix is empty — skipping")
                return
            if confidence < LESSON_MIN_CONFIDENCE:
                logger.debug(f"[ReflectionAgent] Confidence {confidence:.2f} below threshold — skipping")
                return

            # 6. Write lesson to Layer 5 (ProceduralMemory)
            # add_pattern() handles deduplication and reinforcement automatically
            await self.l5.add_pattern(
                trigger    = trigger,
                behavior   = fix,
                context    = context_tag,
                confidence = confidence,
            )

            logger.info(
                f"[ReflectionAgent] Lesson learned | "
                f"type={failure_type} | trigger='{trigger}' | "
                f"fix='{fix}' | confidence={confidence:.2f}"
            )

        except asyncio.TimeoutError:
            logger.warning("[ReflectionAgent] Diagnostic LLM timed out (non-fatal)")
        except Exception as exc:
            logger.warning(f"[ReflectionAgent] Diagnostic failed (non-fatal): {exc}")
        finally:
            self._running = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_history(self, history: List[Dict[str, Any]], max_messages: int = 6) -> str:
        """Format recent session history into readable lines for the diagnostic prompt."""
        lines = []
        for msg in history[-max_messages:]:
            role    = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                prefix = "User" if role == "user" else "Friday"
                lines.append(f"{prefix}: {content.strip()[:300]}")
        return "\n".join(lines) if lines else "(no conversation context available)"

    def _parse_lesson(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse the JSON lesson object from LLM output.
        Handles markdown code fences and leading/trailing noise gracefully.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find the outermost JSON object
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            logger.debug(f"[ReflectionAgent] JSON parse error: {exc}")
            return None
