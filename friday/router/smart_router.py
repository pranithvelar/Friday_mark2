"""
Smart Router — Central Coordinator
=====================================
The single entry point for ALL user queries.

NEW: Every call to route() now does two things simultaneously:
  1. WRITE PATH  — asyncio.create_task(memory_pipeline.process()) fires instantly
                   in the background.  Friday learns from EVERY word you say.
  2. READ PATH   — ContextAssembler.build() runs and produces a ContextBundle
                   containing semantic memory search results, your calendar
                   itinerary, reminders, conflict warnings, and profile data.
                   This bundle is passed to EVERY handler — Simple, Medium, Complex.

Routing flow:
  User Query
    │
    ├── [BACKGROUND] MemoryPipeline.process()  ← fires instantly, never waited on
    │
    ├── ContextAssembler.build()               ← semantic search + calendar + profile
    │
    ▼
  FastIntentClassifier  (Tier 1, <50ms, regex + LLM)
    │
    ├── SIMPLE ────► SimpleHandler(bundle)    (LLM + full context, <500ms)
    ├── MEDIUM ────► MediumHandler(bundle)    (AgentLoop, max 3 tools, <2s)
    └── COMPLEX ───► MultiAgentPlanner(bundle) → ExecutionEngine (background)

Special fast-paths (no classifier needed):
  - "what are you doing?" + active execution → instant status from ExecutionState
  - "what's the progress?" + active execution → instant % summary

Adapter design (future WhatsApp / Telegram / Frontend):
  SmartRouter is platform-agnostic. It receives (query, session_id) and
  returns a standard RouterResponse dict. The caller (terminal_chat.py,
  a WhatsApp webhook handler, a FastAPI endpoint, etc.) formats the
  response for its own channel.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SmartRouter:
    """
    Platform-agnostic router. One instance lives for the lifetime of the process.

    Parameters
    ----------
    classifier        : FastIntentClassifier / LLMRouter
    simple            : SimpleHandler
    medium            : MediumHandler
    planner           : MultiAgentPlanner
    state_manager     : ExecutionStateManager
    agent_loop        : AgentLoop  (fallback for uncovered cases)
    memory_pipeline   : MemoryPipeline  (fires on every input)
    context_assembler : ContextAssembler (builds context bundle for every input)
    """

    def __init__(
        self,
        classifier,
        simple,
        medium,
        planner,
        state_manager,
        agent_loop,
        memory_pipeline=None,
        context_assembler=None,
    ):
        self.classifier        = classifier
        self.simple            = simple
        self.medium            = medium
        self.planner           = planner
        self.state_manager     = state_manager
        self.loop              = agent_loop
        self.memory_pipeline   = memory_pipeline
        self.context_assembler = context_assembler
        self.chronicle         = None   # set via set_chronicle() after construction
        self._live_ctx         = None   # set via set_live_ctx() after construction
        self._event_engine     = None   # set via set_event_engine() after construction
        # Pending plan store: {session_id: ExecutionPlan}
        # A plan lives here from 'awaiting_approval' until user says yes or no.
        self._pending_plans: dict = {}
        # Per-session clarify counter: after 3 consecutive clarify responses,
        # router auto-escalates to complex so Friday stops asking Qs and starts building.
        self._clarify_count: dict = {}  # {session_id: int}

    def set_chronicle(self, chronicle: dict) -> None:
        """Inject project chronicle components after construction."""
        self.chronicle = chronicle

    def set_live_ctx(self, live_ctx) -> None:
        """Inject LiveContextState so plan approval shows in brain awareness."""
        self._live_ctx = live_ctx

    def set_event_engine(self, event_engine) -> None:
        """Inject EventEngine for instant hot-path event detection + conflict check."""
        self._event_engine = event_engine

    # ──────────────────────────────────────────────────────────────────────
    # Main routing method (platform-agnostic)
    # ──────────────────────────────────────────────────────────────────────

    async def route(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Route a user query to the correct handler.

        ALWAYS fires the memory pipeline in background first.
        ALWAYS builds a context bundle before dispatching.
        Every handler receives the full bundle — no route is left context-blind.

        Returns
        -------
        dict with keys:
            text          : str   — human-readable response (for any channel)
            complexity    : str   — "simple" | "medium" | "complex"
            category      : str   — e.g. "calendar_query"
            execution_id  : str   — None unless Complex tier was used
            tools_used    : list  — tool names actually called
            latency_ms    : float — time to first response in milliseconds
            meta          : dict  — any extra data (plan steps, etc.)
        """
        start = time.monotonic()

        # ── 1. WRITE PATH: fire memory indexing as background task ──────────────
        # True fire-and-forget. User NEVER waits. Embedding runs in background.
        if self.memory_pipeline:
            asyncio.create_task(
                self.memory_pipeline.process(query, session_id, role="user")
            )

        # ── 1b. EVENT PATH: instant event detection + conflict check ────────────
        # Runs on the HOT PATH — pure regex + SQLite, zero LLM, ~10ms.
        # Returns conflict data BEFORE the LLM generates a single token.
        conflict_events: list = []
        if self._event_engine:
            try:
                event_result = await self._event_engine.check(query, session_id)
                conflict_events = event_result.conflicts
            except Exception as e:
                logger.warning(f"[Router] EventEngine failed (non-fatal): {e}")

        # ── 1b. PROJECT CHRONICLE: classify + log in background ───────────────
        # SafeChronicle wraps every call: atomic writes, per-project lock,
        # circuit breaker. Chronicle failure NEVER bubbles up to the user.
        _project_question = ""
        _safe = self.chronicle.get("safe") if self.chronicle else None
        if _safe:
            try:
                active_slug    = _safe.get_active_project()
                classification = await _safe.classify(query, active_slug)
                if classification.slug:
                    # Atomic, lock-guarded log write
                    await _safe.log_message(
                        classification.slug, query,
                        role="user", confidence=classification.confidence,
                    )
                    if classification.slug != active_slug:
                        _safe.set_active_project(classification.slug)
                        clf = _safe.classifier
                        if clf:
                            clf.update_active_project(classification.slug)
                if classification.ask_question:
                    _project_question = classification.ask_question
            except Exception as e:
                # Belt-and-suspenders: SafeChronicle should never raise,
                # but if it does the router still works normally.
                logger.warning(f"[Router] Chronicle block failed (safe): {e}")


        # ── 2. READ PATH: build shared context bundle for ALL handlers ─────────
        # Passes conflict_events (from EventEngine) as pre_conflicts so
        # ContextAssembler injects them at the TOP of conflict_ctx before
        # any DB reads — conflict warning is always the first thing LLM sees.
        bundle = None
        if self.context_assembler:
            try:
                bundle = await self.context_assembler.build(
                    query,
                    session_id,
                    pre_conflicts=conflict_events if conflict_events else None,
                )
            except Exception as e:
                logger.warning(f"[Router] ContextAssembler failed: {e}")

        # Expose bundle to route_and_learn() for L4 usefulness tracking
        self._last_bundle = bundle

        # ── Fast-path 0b: PLAN APPROVAL GATE ──────────────────────────────────
        # Intercept YES/NO/REPLACE before LLM classifier runs.
        # _pending_plans stores {"plan": plan, "created_at": monotonic, "query": str}
        pending_entry = self._pending_plans.get(session_id)
        if pending_entry is not None:
            # ── TTL check: auto-expire after 10 minutes ────────────────────
            if time.monotonic() - pending_entry["created_at"] > 600:
                del self._pending_plans[session_id]
                await self._clear_pending_plan_block(session_id)
                logger.info("[Router] Pending plan expired (>10 min) — routing normally")
                pending_entry = None   # fall through to normal routing
            else:
                pending    = pending_entry["plan"]
                orig_query = pending_entry["query"]
                _q_lower   = query.strip().lower()

                _AFFIRM  = {"yes", "yeah", "yep", "yup", "ok", "okay", "proceed",
                            "go", "go ahead", "do it", "fire", "run it", "execute",
                            "confirm", "approved", "start", "sure", "lets go", "let's go"}
                _DENY    = {"no", "nope", "nah", "cancel", "abort", "stop", "don't",
                            "dont", "skip", "never mind", "nevermind", "forget it"}
                _REPLACE = {"replace", "new plan", "different", "change it",
                            "swap", "something else", "actually"}

                if any(_q_lower == w or _q_lower.startswith(w + " ") for w in _AFFIRM):
                    # User approved — fire immediately
                    del self._pending_plans[session_id]
                    await self._clear_pending_plan_block(session_id)
                    result = await self.planner.fire_plan(pending, session_id)
                    return self._build_response(
                        text=result["text"],
                        complexity="complex",
                        category="plan_approved",
                        tools_used=result.get("tools_used", []),
                        execution_id=result.get("execution_id"),
                        latency_ms=(time.monotonic() - start) * 1000,
                        meta={"plan_steps": result.get("plan_steps", 0), "mode": "multi_agent"},
                    )

                elif any(_q_lower == w or _q_lower.startswith(w + " ") for w in _DENY):
                    # User cancelled — clear plan
                    del self._pending_plans[session_id]
                    await self._clear_pending_plan_block(session_id)
                    return self._build_response(
                        text="Understood, Sir. Plan discarded.",
                        complexity="simple",
                        category="plan_cancelled",
                        latency_ms=(time.monotonic() - start) * 1000,
                    )

                elif any(_q_lower == w or _q_lower.startswith(w + " ") for w in _REPLACE):
                    # User wants a new plan instead — discard old and re-route as complex
                    del self._pending_plans[session_id]
                    await self._clear_pending_plan_block(session_id)
                    logger.info("[Router] User replaced pending plan — generating new plan")
                    return await self._route_complex(query, session_id, start, bundle)

                else:
                    # User said something else while plan is pending — remind them
                    pending_reminder = (
                        f"Sir, you have a pending plan for: \"{orig_query[:60]}\"\n"
                        "Say **yes** to execute, **cancel** to discard, "
                        "or **replace** to plan a different request."
                    )
                    if bundle is not None:
                        bundle.conflict_ctx = (
                            pending_reminder + "\n" + (bundle.conflict_ctx or "")
                        ).strip()

        # ── Fast-path 1: Progress query with active execution ──────────────────
        active_exec = self.state_manager.get_session_execution(session_id)
        if active_exec:
            from friday.router.intent_classifier import QueryCategory
            complexity, category = await self.classifier.classify(query, session_id)
            if category == QueryCategory.PROGRESS_QUERY:
                text = active_exec.format_progress_response()
                return self._build_response(
                    text=text,
                    complexity="simple",
                    category="progress_query",
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            # User sent something while engine is running → treat as interrupt
            active_exec.request_interrupt(query)
            return self._build_response(
                text=(
                    f"Understood, Sir. I will adjust the current execution accordingly. "
                    f"Currently at {active_exec.progress_percent:.0f}% progress."
                ),
                complexity="simple",
                category="execution_interrupt",
                latency_ms=(time.monotonic() - start) * 1000,
            )

        # ── 3. Classify via LLM (with regex fallback) ──────────────────────────
        complexity, category = await self.classifier.classify(query, session_id)
        logger.info(
            f"[Router] query='{query[:60]}' "
            f"→ complexity={complexity.value} category={category.value}"
        )

        # ── 3b. CLARIFY fast-path ──────────────────────────────────────
        # Router returned 'clarify' — request is complex but too vague to plan.
        # Enforce MAX 3 clarify questions per session window.
        # After 3, auto-escalate to complex so Friday stops asking and starts building.
        from friday.router.intent_classifier import QueryCategory
        if category == QueryCategory.CLARIFY:
            count = self._clarify_count.get(session_id, 0) + 1
            self._clarify_count[session_id] = count
            if count >= 3:
                # Auto-escalate: reset counter, route as complex
                logger.info(
                    f"[Router] Clarify limit reached ({count}/3) for session '{session_id}' "
                    "— auto-escalating to complex"
                )
                self._clarify_count[session_id] = 0
                return await self._route_complex(query, session_id, start, bundle)
            return await self._route_clarify(query, session_id, start, bundle)

        # ── 4. Route by complexity — ALL handlers receive the context bundle ───
        try:
            if complexity.value == "simple":
                # Non-clarify response: reset clarify counter
                self._clarify_count[session_id] = 0
                result = await self._route_simple(query, category, session_id, start, bundle)

            elif complexity.value == "medium":
                self._clarify_count[session_id] = 0
                result = await self._route_medium(query, session_id, start, bundle)

            else:  # complex
                self._clarify_count[session_id] = 0
                result = await self._route_complex(query, session_id, start, bundle)

            # ── 4b. Append project clarification question if classifier was uncertain
            # Only ONE question ever appended — non-intrusive, at the end of response.
            if _project_question and result.get("text"):
                result["text"] = result["text"].rstrip() + "\n\n" + _project_question
            return result

        except asyncio.TimeoutError:
            logger.error(f"[Router] Timeout routing query: {query[:60]}")
            return self._build_response(
                text="I apologise, Sir — that request timed out. Please try again.",
                complexity=complexity.value,
                category=category.value,
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            logger.exception(f"[Router] Unhandled error: {e}")
            return self._build_response(
                text=f"I encountered an unexpected error, Sir: {e}",
                complexity=complexity.value,
                category=category.value,
                latency_ms=(time.monotonic() - start) * 1000,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Tier handlers — all accept context bundle
    # ──────────────────────────────────────────────────────────────────────

    async def _route_clarify(
        self, query: str, session_id: str,
        start: float, bundle=None
    ) -> Dict[str, Any]:
        """
        Called when LLM router returns 'clarify'.
        Generates ONE targeted question via a tiny LLM call.
        No plan. No execution. Returns immediately.
        """
        clarify_text = "Could you give me a bit more detail about that, Sir?"
        try:
            if self.loop:
                messages = [
                    {"role": "system", "content": (
                        "You are Friday, a concise personal AI assistant. "
                        "The user's request is too vague to act on. "
                        "Ask them ONE specific, targeted question to get the "
                        "clarity you need. Keep it to ONE sentence maximum. "
                        "Be direct. Address them as Sir."
                    )},
                    {"role": "user", "content": query},
                ]
                response = await self.loop._llm_call(messages)
                if response and response.strip():
                    clarify_text = response.strip()
        except Exception as e:
            logger.warning(f"[Router] Clarify LLM call failed: {e}")

        return self._build_response(
            text=clarify_text,
            complexity="simple",
            category="clarify",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def _route_simple(
        self, query: str, category, session_id: str,
        start: float, bundle=None
    ) -> Dict[str, Any]:
        text = await self.simple.handle(query, category, session_id, bundle)
        return self._build_response(
            text=text,
            complexity="simple",
            category=category.value,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def _route_medium(
        self, query: str, session_id: str,
        start: float, bundle=None
    ) -> Dict[str, Any]:
        result = await self.medium.handle(query, session_id, bundle)
        return self._build_response(
            text=result["text"],
            complexity="medium",
            category="medium",
            tools_used=result.get("tools_used", []),
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def _route_complex(
        self, query: str, session_id: str,
        start: float, bundle=None
    ) -> Dict[str, Any]:
        # ── Replace-guard: if plan is still pending, block silent overwrite ────
        existing = self._pending_plans.get(session_id)
        if existing and isinstance(existing, dict):
            orig = existing.get("query", "")[:60]
            return self._build_response(
                text=(
                    f"Sir, you still have a pending plan for: \"{orig}\"\n"
                    "Say **yes** to run it, **cancel** to discard it, "
                    "or **replace** to generate a new plan for this request."
                ),
                complexity="complex",
                category="plan_conflict",
                latency_ms=(time.monotonic() - start) * 1000,
            )

        result = await self.planner.execute(query, session_id, bundle)

        # If planner returned an approval-pending plan, store it with TTL metadata
        if result.get("mode") == "awaiting_approval":
            pending_plan = result.pop("pending_plan", None)
            if pending_plan:
                self._pending_plans[session_id] = {
                    "plan":       pending_plan,
                    "created_at": time.monotonic(),
                    "query":      query,
                }
                await self._set_pending_plan_block(session_id, result["text"])

        return self._build_response(
            text=result["text"],
            complexity="complex",
            category=result.get("category", "complex"),
            tools_used=result.get("tools_used", []),
            execution_id=result.get("execution_id"),
            latency_ms=(time.monotonic() - start) * 1000,
            meta={
                "plan_steps": result.get("plan_steps", 0),
                "mode":       result.get("mode", "multi_agent"),
            },
        )

    async def _set_pending_plan_block(self, session_id: str, plan_text: str):
        """Inject pending plan into LiveContextState so brain sees it always."""
        if self._live_ctx is None:
            return
        try:
            await self._live_ctx.update(
                current_time_str   = self._live_ctx.current_time_str,
                itinerary_block    = self._live_ctx.itinerary_block,
                conflict_block     = self._live_ctx.conflict_block,
                reminder_block     = self._live_ctx.reminder_block,
                execution_block    = self._live_ctx.execution_block,
                pending_plan_block = (
                    f"[PENDING PLAN — AWAITING USER APPROVAL]\n{plan_text}\n"
                    f"Tell the user to say 'yes' to execute or 'cancel' to discard."
                ),
            )
        except Exception as e:
            logger.debug(f"[Router] pending_plan_block update failed: {e}")

    async def _clear_pending_plan_block(self, session_id: str):
        """Clear pending plan from LiveContextState after approval/rejection."""
        if self._live_ctx is None:
            return
        try:
            await self._live_ctx.update(
                current_time_str   = self._live_ctx.current_time_str,
                itinerary_block    = self._live_ctx.itinerary_block,
                conflict_block     = self._live_ctx.conflict_block,
                reminder_block     = self._live_ctx.reminder_block,
                execution_block    = self._live_ctx.execution_block,
                pending_plan_block = "",
            )
        except Exception as e:
            logger.debug(f"[Router] clear pending_plan_block failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Also fire learning pipeline on bot responses so FRIDAY's own words
    # become searchable in future queries
    # ──────────────────────────────────────────────────────────────────────

    async def route_and_learn(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Convenience wrapper: route the query AND index the bot's reply.
        Also passively reinforces L4 facts that were injected and led to
        a non-trivial response — survival of the fittest for semantic memory.
        """
        result = await self.route(query, session_id)
        response_text = result.get("text", "")

        if self.memory_pipeline and response_text:
            # Index assistant's own response so it's searchable later
            asyncio.create_task(
                self.memory_pipeline.process(
                    response_text, session_id, role="assistant"
                )
            )

        # ── Passive L4 usefulness signal ────────────────────────────────────
        # If FRIDAY gave a real response (>20 words, not "I don't know"),
        # the facts injected this turn probably helped — boost their confidence.
        # Facts that are never useful decay naturally via temporal_decay.
        _last_bundle = getattr(self, "_last_bundle", None)
        if (
            _last_bundle is not None
            and _last_bundle.injected_fact_ids
            and len(response_text.split()) > 20
            and self.context_assembler
            and self.context_assembler.semantic_memory
        ):
            for fid in _last_bundle.injected_fact_ids:
                try:
                    self.context_assembler.semantic_memory.record_usefulness(fid)
                except Exception:
                    pass

        return result


    # ──────────────────────────────────────────────────────────────────────
    # Standard response builder
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_response(
        text: str,
        complexity: str,
        category: str,
        tools_used: Optional[list] = None,
        execution_id: Optional[str] = None,
        latency_ms: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "text":         text,
            "complexity":   complexity,
            "category":     category,
            "tools_used":   tools_used or [],
            "execution_id": execution_id,
            "latency_ms":   round(latency_ms, 2),
            "meta":         meta or {},
        }


# ── Factory helper ────────────────────────────────────────────────────────────

def build_smart_router(
    agent_loop,
    db_manager,
    personalization,
    searcher,
    llm_provider=None,
    memory_pipeline=None,
    context_assembler=None,
) -> SmartRouter:
    """
    Convenience factory — assembles all components and returns a SmartRouter.

    Parameters
    ----------
    memory_pipeline   : MemoryPipeline   — created in chat.py, passed here
    context_assembler : ContextAssembler — created in chat.py, passed here

    Example
    -------
    router = build_smart_router(
        loop, db_manager, personalization, searcher,
        memory_pipeline=pipeline,
        context_assembler=assembler,
    )
    result = await router.route_and_learn(user_message, SESSION_ID)
    print(result["text"])
    """
    from friday.router.intent_classifier import FastIntentClassifier
    from friday.router.llm_router import LLMRouter
    from friday.router.handlers.simple_handler import SimpleHandler
    from friday.router.handlers.medium_handler import MediumHandler
    from friday.router.handlers.complex_handler import MultiAgentPlanner
    from friday.execution.subagent_registry import SubagentRegistry
    from friday.execution.state_manager import ExecutionStateManager
    from friday.execution.engine import ExecutionEngine
    from friday.execution.memory_aware_executor import MemoryAwareExecutor
    from friday.execution.learning import LearningEngine

    # Build components bottom-up
    learning        = LearningEngine(db_manager)
    memory_exec     = MemoryAwareExecutor(personalization, searcher, learning)
    state_manager   = ExecutionStateManager()
    engine          = ExecutionEngine(agent_loop, state_manager, memory_exec, learning)
    subagent_reg    = SubagentRegistry()

    planner = MultiAgentPlanner(
        agent_loop=agent_loop,
        db_manager=db_manager,
        personalization=personalization,
        execution_engine=engine,
        subagent_registry=subagent_reg,
    )

    # LLMRouter wraps the regex classifier as its fallback
    regex_fallback = FastIntentClassifier()
    llm_router     = LLMRouter(
        model=agent_loop.model,
        fallback_classifier=regex_fallback,
        llm_provider=llm_provider,
    )

    return SmartRouter(
        classifier        = llm_router,
        simple            = SimpleHandler(db_manager, personalization, searcher),
        medium            = MediumHandler(agent_loop),
        planner           = planner,
        state_manager     = state_manager,
        agent_loop        = agent_loop,
        memory_pipeline   = memory_pipeline,
        context_assembler = context_assembler,
    )
