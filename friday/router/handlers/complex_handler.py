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

    async def execute(self, query: str, session_id: str, bundle=None) -> Dict[str, Any]:
        """
        PHASE 1 of execution: generate plan and return it for user approval.
        FRIDAY presents the plan and waits. Execution is NOT triggered here.

        After user approves (SmartRouter detects 'yes/proceed'),
        SmartRouter calls fire_plan() with the stored plan.

        Returns dict with mode='awaiting_approval' and the human-readable plan.
        """
        self.loop._status("Thinking through your request...")

        # Use pre-built bundle from SmartRouter, or gather fresh if not provided
        if bundle is not None and not bundle.is_empty:
            memory_ctx = bundle.augmented_prefix
        else:
            memory_ctx = await self._gather_memory_context(query, session_id)

        # Generate plan via LLM
        plan = await self._generate_plan(query, memory_ctx)
        if not plan:
            # Plan generation failed (e.g. LLM rate-limit, timeout, or bad parse).
            # Do NOT fall back to raw AgentLoop — it will loop on tool calls and hit max_steps.
            # Instead, respond conversationally so the user can rephrase or retry.
            logger.warning("[Planner] Plan generation failed — responding gracefully")
            # Try a simple one-shot LLM response with no tools as a best-effort answer
            try:
                graceful = await self.loop._llm_call([
                    {"role": "system", "content": (
                        "You are Friday, a concise personal AI assistant. "
                        "Acknowledge the user's idea or request in 1-2 sentences, "
                        "then ask ONE clarifying question to help you plan the next step. "
                        "Be direct. Address them as Sir."
                    )},
                    {"role": "user", "content": query},
                ])
                if graceful and graceful.strip():
                    return {
                        "text": graceful.strip(),
                        "tools_used": [],
                        "execution_id": None,
                        "plan_steps": 0,
                        "mode": "clarify_fallback",
                    }
            except Exception as e:
                logger.warning(f"[Planner] Graceful fallback LLM call failed: {e}")
            return {
                "text": (
                    "That sounds interesting, Sir. Could you give me a bit more detail "
                    "so I can put together a solid plan for you?"
                ),
                "tools_used": [],
                "execution_id": None,
                "plan_steps": 0,
                "mode": "clarify_fallback",
            }

        # Format plan as JARVIS-style approval request
        plan_text = self._format_plan_for_approval(query, plan)

        logger.info(f"[Planner] Plan ready ({len(plan.steps)} steps) — awaiting approval")
        return {
            "text": plan_text,
            "tools_used": [],
            "execution_id": None,
            "plan_steps": len(plan.steps),
            "mode": "awaiting_approval",
            "pending_plan": plan,   # SmartRouter stores this, fires after 'yes'
        }

    async def fire_plan(self, plan, session_id: str) -> Dict[str, Any]:
        """
        PHASE 2: called by SmartRouter when user approves the plan.
        Submits the pre-built plan to the ExecutionEngine immediately.
        """
        execution_id = await self.execution_engine.execute_plan(plan, session_id)
        step_count = len(plan.steps)
        self.loop._status(f"Executing {step_count}-step plan...")
        logger.info(f"[Planner] Fired plan {plan.plan_id[:8]} — {step_count} steps")
        return {
            "text": (
                f"On it, Sir. Executing {step_count} steps in the background. "
                "I'll update you when each step completes."
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

    def _format_plan_for_approval(self, query: str, plan) -> str:
        """
        JARVIS-style plan presentation.
        Shows the user exactly what will happen before a single tool fires.
        """
        est_total = plan.estimated_duration_seconds
        if est_total < 60:
            est_str = f"~{est_total}s"
        else:
            est_str = f"~{est_total // 60}m {est_total % 60}s"

        lines = [
            f"Here's my plan for: *{query[:80]}*",
            "",
        ]
        for step in plan.steps:
            tag = f"[{step.tool_category.upper()}]"
            lines.append(f"  {step.step_number}. {tag} {step.action}")

        lines.extend([
            "",
            f"Estimated time: {est_str}  |  {len(plan.steps)} steps",
            "",
            "Shall I proceed, Sir?",
        ])
        return "\n".join(lines)

    def _format_plan_preview(self, plan) -> str:
        lines = []
        for step in plan.steps:
            lines.append(
                f"  Step {step.step_number}: [{step.tool_category.upper()}] {step.action}"
            )
        return "\n".join(lines)
