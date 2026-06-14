"""
Multi-Agent Planner (Tier 2C — Complex)
=========================================
OpenClaw-inspired multi-step planner.

How it works:
  1. LLM generates a markdown-fenced plan (```json inside markdown, same
     extraction pattern as loop.py's _extract_action).
  2. Plan is parsed into ExecutionPlan / ExecutionStep objects.
  3. Memory context is injected BEFORE plan generation.
  4. ExecutionEngine runs the plan independently.

Plan JSON format (extracted from markdown code fence):
  ```json
  [
    {
      "step": 1,
      "action": "Search Google for latest AI papers",
      "tool_category": "browser",
      "reasoning": "Need to gather up-to-date sources",
      "estimated_seconds": 15
    },
    ...
  ]
  ```
"""

import json
import re
import uuid
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from friday.execution.state_manager import ExecutionPlan, ExecutionStep

logger = logging.getLogger(__name__)

# ── Planner system prompt ─────────────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = """You are a senior execution planner for Friday, an intelligent AI assistant.

Your job is to break complex user tasks into a concrete step-by-step execution plan.

OUTPUT FORMAT — respond with ONLY a markdown JSON code block:
```json
[
  {
    "step": 1,
    "action": "Describe what to do in plain English",
    "tool_category": "browser | memory | search | bash | mcp | none",
    "reasoning": "Why this step is needed",
    "estimated_seconds": 10
  }
]
```

RULES:
1. Maximum 10 steps. If task needs more, split into sub-tasks.
2. Each step must be atomic (one thing only).
3. tool_category must be one of: browser, memory, search, bash, mcp, none
4. reasoning must explain WHY, not WHAT.
5. Do NOT include any text outside the ```json block.
6. Steps should be ordered logically — dependencies first.
"""


class MultiAgentPlanner:
    """
    Tier 2C: Complex multi-step planner.
    Uses the same AgentLoop._llm_call() and _extract_action() patterns
    as the existing system — no new LLM infrastructure.
    """

    def __init__(self, agent_loop, db_manager, personalization, execution_engine, subagent_registry=None):
        self.loop = agent_loop
        self.db = db_manager
        self.personalization = personalization
        self.execution_engine = execution_engine
        self.subagent_registry = subagent_registry

    # ── Main entry point ──────────────────────────────────────────────────

    async def execute(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Build a plan then hand it to the ExecutionEngine.
        Returns immediately with execution_id — engine runs in background.
        """
        self.loop._status("Planning multi-step task...")

        # 1. Build memory context to inject into planning
        memory_ctx = await self._gather_memory_context(query, session_id)

        # 2. Generate plan via LLM
        plan = await self._generate_plan(query, memory_ctx)
        if not plan:
            # Fallback: let AgentLoop handle it directly
            logger.info("[Planner] Plan generation failed; falling back to AgentLoop")
            response = await self.loop.run(query, max_steps=5)
            return {
                "text": response,
                "tools_used": [],
                "execution_id": None,
                "plan_steps": 0,
                "mode": "fallback",
            }

        # 3. Register with execution engine
        execution_id = await self.execution_engine.execute_plan(plan, session_id)

        step_count = len(plan.steps)
        self.loop._status(f"Plan ready: {step_count} steps. Execution started.")

        return {
            "text": (
                f"Understood, Sir. On it."
            ),
            "tools_used": [s.tool_category for s in plan.steps],
            "execution_id": execution_id,
            "plan_steps": step_count,
            "mode": "multi_agent",
        }

    # ── Plan generation ───────────────────────────────────────────────────

    async def _generate_plan(self, query: str, memory_ctx: str):
        """Ask LLM to produce a markdown-fenced JSON plan."""
        from friday.execution.state_manager import ExecutionPlan, ExecutionStep

        planning_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]

        # Inject memory context if available
        user_content = query
        if memory_ctx:
            user_content = f"[Context from memory]\n{memory_ctx}\n\n[Task]\n{query}"

        planning_messages.append({"role": "user", "content": user_content})

        try:
            raw = await self.loop._llm_call(planning_messages)
        except asyncio.TimeoutError:
            logger.error("[Planner] LLM call timed out during plan generation")
            return None
        except Exception as e:
            logger.error(f"[Planner] LLM call failed: {e}")
            return None

        steps_data = self._extract_plan_from_markdown(raw)
        if not steps_data:
            logger.debug(f"[Planner] Could not parse plan from LLM output:\n{raw[:300]}")
            return None

        # Build typed ExecutionStep list
        steps: List[ExecutionStep] = []
        total_est = 0
        for i, s in enumerate(steps_data):
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                step_number=i + 1,
                action=s.get("action", f"Step {i+1}"),
                tool_category=s.get("tool_category", "none"),
                reasoning=s.get("reasoning", ""),
                estimated_seconds=int(s.get("estimated_seconds", 10)),
            )
            steps.append(step)
            total_est += step.estimated_seconds

        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            query=query,
            steps=steps,
            estimated_duration_seconds=total_est,
            complexity="complex",
            created_at=datetime.now(),
        )
        logger.info(f"[Planner] Generated plan: {len(steps)} steps, ~{total_est}s")
        return plan

    def _extract_plan_from_markdown(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extracts JSON plan from markdown code fence.
        Uses the SAME pattern as loop.py _extract_action() — markdown first,
        then raw JSON array fallback.
        """
        # Primary: ```json ... ``` fence
        if "```json" in text:
            try:
                json_str = text.split("```json")[-1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        # Fallback: raw JSON array in text
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group())
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        return None

    # ── Memory context gathering ──────────────────────────────────────────

    async def _gather_memory_context(self, query: str, session_id: str) -> str:
        """
        Pre-query memory BEFORE generating the plan.
        Uses hybrid search (same as loop.py _pre_search).
        Also pulls user preferences for personalization of the plan.
        """
        ctx_parts = []

        # 1. Semantic memory search
        if self.loop._searcher:
            try:
                results = await self.loop._searcher.search(
                    query, vector_weight=0.6, text_weight=0.4, max_results=3
                )
                if results:
                    snippets = [r.snippet[:150] for r in results if r.snippet]
                    if snippets:
                        ctx_parts.append("Relevant memory:\n" + "\n".join(f"- {s}" for s in snippets))
            except Exception as e:
                logger.debug(f"[Planner] Memory search failed: {e}")

        # 2. User preferences
        if self.personalization:
            pref_ctx = self.personalization.get_context_string()
            if pref_ctx:
                ctx_parts.append(pref_ctx)

        return "\n\n".join(ctx_parts)

    # ── Formatting ────────────────────────────────────────────────────────

    def _format_plan_preview(self, plan) -> str:
        lines = []
        for step in plan.steps:
            lines.append(
                f"  Step {step.step_number}: [{step.tool_category.upper()}] {step.action}"
            )
        return "\n".join(lines)
