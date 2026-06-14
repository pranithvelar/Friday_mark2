"""
Smart Router — Central Coordinator
=====================================
The single entry point for ALL user queries.

Routing flow:
  User Query
    │
    ▼
  FastIntentClassifier  (Tier 1, <50ms, pure regex)
    │
    ├── SIMPLE ────► SimpleHandler   (direct DB, <200ms)
    ├── MEDIUM ────► MediumHandler   (AgentLoop, max 3 tools, <2s)
    └── COMPLEX ───► MultiAgentPlanner → ExecutionEngine (background)

Special fast-paths (no classifier needed):
  - "what are you doing?" + active execution → instant status from ExecutionState
  - "what's the progress?" + active execution → instant % summary

Adapter design (future WhatsApp / Telegram / Frontend):
  SmartRouter is platform-agnostic. It receives (query, session_id) and
  returns a standard RouterResponse dict. The caller (terminal_chat.py,
  a WhatsApp webhook handler, a FastAPI endpoint, etc.) formats the
  response for its own channel.

  To add a new channel integration:
    1. Create a new entry script (e.g. whatsapp_adapter.py).
    2. Import SmartRouter, call await router.route(message, session_id).
    3. Format router.response["text"] for WhatsApp.
    No changes to SmartRouter are needed.
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
    classifier  : FastIntentClassifier
    simple      : SimpleHandler
    medium      : MediumHandler
    planner     : MultiAgentPlanner
    state_manager : ExecutionStateManager
    agent_loop  : AgentLoop  (fallback for uncovered cases)
    """

    def __init__(
        self,
        classifier,
        simple,
        medium,
        planner,
        state_manager,
        agent_loop,
    ):
        self.classifier    = classifier
        self.simple        = simple
        self.medium        = medium
        self.planner       = planner
        self.state_manager = state_manager
        self.loop          = agent_loop

    # ──────────────────────────────────────────────────────────────────────
    # Main routing method (platform-agnostic)
    # ──────────────────────────────────────────────────────────────────────

    async def route(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Route a user query to the correct handler.

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

        # ── Fast-path 0: Progress query with active execution ──────────────
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
            if not category == QueryCategory.PROGRESS_QUERY:
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

        # ── Classify via LLM (with regex fallback) ─────────────────────────
        complexity, category = await self.classifier.classify(query, session_id)
        logger.info(
            f"[Router] query='{query[:60]}' "
            f"→ complexity={complexity.value} category={category.value}"
        )

        # ── Route by complexity ────────────────────────────────────────────
        try:
            if complexity.value == "simple":
                return await self._route_simple(query, category, session_id, start)

            elif complexity.value == "medium":
                return await self._route_medium(query, session_id, start)

            else:  # complex
                return await self._route_complex(query, session_id, start)

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
    # Tier handlers
    # ──────────────────────────────────────────────────────────────────────

    async def _route_simple(self, query: str, category, session_id: str, start: float) -> Dict[str, Any]:
        text = await self.simple.handle(query, category, session_id)
        return self._build_response(
            text=text,
            complexity="simple",
            category=category.value,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def _route_medium(self, query: str, session_id: str, start: float) -> Dict[str, Any]:
        result = await self.medium.handle(query, session_id)
        return self._build_response(
            text=result["text"],
            complexity="medium",
            category="medium",
            tools_used=result.get("tools_used", []),
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def _route_complex(self, query: str, session_id: str, start: float) -> Dict[str, Any]:
        result = await self.planner.execute(query, session_id)
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
) -> SmartRouter:
    """
    Convenience factory — assembles all components and returns a SmartRouter.
    Now uses LLMRouter (with regex fallback) instead of hardcoded patterns.

    Example
    -------
    router = build_smart_router(loop, db_manager, personalization, searcher)
    result = await router.route(user_message, SESSION_ID)
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
        classifier    = llm_router,
        simple        = SimpleHandler(db_manager, personalization, searcher),
        medium        = MediumHandler(agent_loop),
        planner       = planner,
        state_manager = state_manager,
        agent_loop    = agent_loop,
    )
