"""
Execution Engine
=================
The autonomous execution loop that runs independently from the LLM conversation.

Key behaviours:
  - Runs as asyncio background task (non-blocking)
  - Real-time state updates via ExecutionState
  - Per-step retry with exponential backoff (up to 3×)
  - Mid-flight plan modification via interrupt
  - Replan on step failure (up to 2 replan attempts)
  - Calls MemoryAwareExecutor before each step
  - Calls LearningEngine after each step (success or failure)

Tool calls are STUBBED. When you integrate real tools, replace
_execute_tool_stub() with your actual tool dispatcher.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from friday.execution.state_manager import ExecutionPlan, ExecutionState, ExecutionStep
from friday.execution.state_manager import ExecutionStatus, ExecutionStateManager, ExecutionPlan

logger = logging.getLogger(__name__)

# ── Safety limits ────────────────────────────────────────────────────────────
MAX_STEP_RETRIES   = 3
MAX_REPLAN_ATTEMPTS = 2
RETRY_BASE_DELAY    = 1.0   # seconds; doubles each attempt

# ── Replan system prompt ──────────────────────────────────────────────────────
REPLAN_SYSTEM_PROMPT = """You are a senior execution planner. A step in the current plan has failed.
Generate a revised plan for the REMAINING steps only.

OUTPUT FORMAT:
```json
[
  {
    "step": <number>,
    "action": "<description>",
    "tool_category": "browser | search | bash | mcp | memory | none",
    "reasoning": "<why>",
    "estimated_seconds": <int>
  }
]
```

Do NOT include already-completed steps. Output ONLY the JSON block."""


class ExecutionEngine:
    """
    Background async execution engine.

    Created once in terminal_chat.py, shared by SmartRouter → MultiAgentPlanner.
    """

    def __init__(self, agent_loop, state_manager: ExecutionStateManager,
                 memory_aware_executor=None, learning_engine=None):
        self.loop              = agent_loop
        self.state_manager     = state_manager
        self.memory_exec       = memory_aware_executor
        self.learning          = learning_engine
        self._replan_count     = 0  # per-execution replan counter (reset per execution)

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    async def execute_plan(self, plan: ExecutionPlan, session_id: str) -> str:
        """
        Start execution of a plan in the background.
        Returns execution_id immediately — caller does not wait.
        """
        state = self.state_manager.create_execution(plan, session_id)
        self._replan_count = 0
        # Fire and forget — the engine runs independently
        asyncio.create_task(self._execute_async(state))
        return state.execution_id

    # ──────────────────────────────────────────────────────────────────────
    # Background execution loop
    # ──────────────────────────────────────────────────────────────────────

    async def _execute_async(self, state):
        """
        The main execution loop. Runs in the background.
        Friday's conversation stays fully responsive during this.
        """
        from friday.execution.state_manager import ExecutionStatus

        state.start()
        logger.info(f"[Engine] Starting execution {state.execution_id[:8]}: "
                    f"{len(state.plan.steps)} steps")

        try:
            while state.current_step_index < len(state.plan.steps):

                # ── Check for interrupt (mid-flight modification) ──────────
                if state.interrupt_requested:
                    await self._handle_interrupt(state)
                    state.clear_interrupt()
                    # After handling interrupt, continue from current position
                    continue

                current_step = state.get_current_step()
                if not current_step:
                    break

                logger.info(
                    f"[Engine] Step {state.current_step_index + 1}/{len(state.plan.steps)}: "
                    f"{current_step.action[:60]}"
                )

                # ── Pre-step memory check ─────────────────────────────────
                if self.memory_exec:
                    try:
                        mem_result = await self.memory_exec.pre_step_check(current_step, state.session_id)
                        if mem_result.suggestions:
                            state.log_action("memory_suggestion", {"suggestions": mem_result.suggestions})
                    except Exception as mem_err:
                        logger.debug(f"[Engine] Memory pre-check failed (non-fatal): {mem_err}")


                # ── Execute step with retry ───────────────────────────────
                success = await self._execute_step_with_retry(state, current_step)

                if not success:
                    # Step failed after all retries → try to replan
                    if self._replan_count < MAX_REPLAN_ATTEMPTS:
                        replanned = await self._attempt_replan(state, current_step)
                        self._replan_count += 1
                        if replanned:
                            logger.info(f"[Engine] Replan #{self._replan_count} succeeded")
                            continue  # restart loop with new plan from same index
                    # All recovery options exhausted
                    self.state_manager.fail_execution(
                        state.execution_id,
                        f"Step {current_step.step_number} failed after retries and replan"
                    )
                    return

                # ── Advance to next step ──────────────────────────────────
                state.advance_step()

            # ── All steps completed ───────────────────────────────────────
            self.state_manager.complete_execution(state.execution_id)
            logger.info(f"[Engine] Execution {state.execution_id[:8]} COMPLETE")

        except asyncio.CancelledError:
            logger.warning(f"[Engine] Execution {state.execution_id[:8]} cancelled")
            self.state_manager.fail_execution(state.execution_id, "Cancelled")
        except Exception as e:
            logger.error(f"[Engine] Unexpected error in execution {state.execution_id[:8]}: {e}")
            self.state_manager.fail_execution(state.execution_id, str(e))

    # ──────────────────────────────────────────────────────────────────────
    # Step execution with retry
    # ──────────────────────────────────────────────────────────────────────

    async def _execute_step_with_retry(self, state, step) -> bool:
        """
        Execute a single step with exponential-backoff retry.
        Returns True on success, False if all retries exhausted.
        """
        from friday.execution.state_manager import ExecutionStatus

        for attempt in range(1, MAX_STEP_RETRIES + 1):
            try:
                state.update_step_status(step.step_id, ExecutionStatus.RUNNING)
                state.log_action(
                    f"Executing step {step.step_number}: {step.action}",
                    {"tool_category": step.tool_category, "attempt": attempt},
                )

                # ── Execute tool (stubbed until tools are integrated) ─────
                result = await self._execute_tool_stub(step)

                # ── Success ───────────────────────────────────────────────
                state.update_step_status(step.step_id, ExecutionStatus.COMPLETE, result=result)
                state.log_action(
                    f"Completed step {step.step_number}",
                    {"result_preview": str(result)[:120]},
                )

                # ── Record success pattern ────────────────────────────────
                if self.learning:
                    await self.learning.record_acceptance(
                        pattern_type="step_approach",
                        context=state.plan.query[:80],
                        key=step.action[:80],
                        value=step.tool_category,
                    )

                return True

            except asyncio.TimeoutError:
                err_msg = f"Step {step.step_number} timed out (attempt {attempt}/{MAX_STEP_RETRIES})"
                logger.warning(f"[Engine] {err_msg}")
                state.record_error(err_msg, step.step_id, {"attempt": attempt})
                step.retry_count = attempt

            except Exception as e:
                err_msg = f"Step {step.step_number} error: {e} (attempt {attempt}/{MAX_STEP_RETRIES})"
                logger.warning(f"[Engine] {err_msg}")
                state.record_error(str(e), step.step_id, {"attempt": attempt})
                step.retry_count = attempt

                # Record failure pattern for learning
                if self.learning:
                    await self.learning.record_correction(
                        pattern_type="step_approach",
                        context=state.plan.query[:80],
                        key=step.action[:80],
                        old_value=step.tool_category,
                        new_value="unknown",
                    )

            if attempt < MAX_STEP_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                state.status = ExecutionStatus.RETRYING
                state.log_action(f"Retrying step {step.step_number}", {"delay_seconds": delay, "attempt": attempt + 1})
                await asyncio.sleep(delay)

        state.update_step_status(
            step.step_id, ExecutionStatus.FAILED,
            error=f"Failed after {MAX_STEP_RETRIES} attempts"
        )
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Tool execution (STUB — replace when integrating real tools)
    # ──────────────────────────────────────────────────────────────────────

    async def _execute_tool_stub(self, step) -> Dict[str, Any]:
        """
        STUB: Simulates tool execution.
        When you integrate real tools, replace this with a real tool dispatcher
        that routes based on step.tool_category:
          "browser"  → Playwright / Selenium tool
          "search"   → HybridSearcher or web search API
          "bash"     → shell command executor
          "mcp"      → MCP protocol client
          "memory"   → direct DB query
          "none"     → LLM reasoning only
        """
        # Small delay to simulate real work
        await asyncio.sleep(0.1)
        return {
            "status":  "stub",
            "step":    step.step_number,
            "action":  step.action,
            "tool":    step.tool_category,
            "message": f"[Tool stub] Would execute '{step.action}' via {step.tool_category}",
        }

    # ──────────────────────────────────────────────────────────────────────
    # Replan on failure
    # ──────────────────────────────────────────────────────────────────────

    async def _attempt_replan(self, state, failed_step) -> bool:
        """
        Ask the LLM to generate a revised plan for remaining steps.
        Replaces state.plan.steps[current_index:] with new steps.
        Returns True if replan succeeded.
        """
        from friday.execution.state_manager import ExecutionStatus

        state.status = ExecutionStatus.REPLANNING
        state.log_action("Replanning after failure", {
            "failed_step": failed_step.step_number,
            "failed_action": failed_step.action,
        })

        completed_summary = self._summarise_completed_steps(state)
        remaining_count = len(state.plan.steps) - state.current_step_index

        replan_prompt = (
            f"Original task: {state.plan.query}\n\n"
            f"Completed so far:\n{completed_summary}\n\n"
            f"FAILED step:\n"
            f"  Action: {failed_step.action}\n"
            f"  Tool: {failed_step.tool_category}\n"
            f"  Error: {failed_step.error or 'unknown'}\n\n"
            f"Generate {remaining_count} revised steps to complete the task differently."
        )

        try:
            raw = await self.loop._llm_call([
                {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
                {"role": "user",   "content": replan_prompt},
            ])
        except Exception as e:
            logger.error(f"[Engine] Replan LLM call failed: {e}")
            return False

        new_steps = self._parse_plan_json(raw)
        if not new_steps:
            logger.warning("[Engine] Replan produced no parseable steps")
            return False

        from friday.execution.state_manager import ExecutionStep
        replacement = []
        base_num = state.current_step_index + 1
        for i, s in enumerate(new_steps):
            replacement.append(ExecutionStep(
                step_id=str(uuid.uuid4()),
                step_number=base_num + i,
                action=s.get("action", f"Step {base_num + i}"),
                tool_category=s.get("tool_category", "none"),
                reasoning=s.get("reasoning", "Replanned step"),
                estimated_seconds=int(s.get("estimated_seconds", 10)),
            ))

        # Replace remaining steps in-place
        state.plan.steps = state.plan.steps[:state.current_step_index] + replacement
        state.status = ExecutionStatus.RUNNING
        state.log_action("Replanned successfully", {"new_steps": len(replacement)})
        return True

    def _summarise_completed_steps(self, state) -> str:
        completed = [
            s for s in state.plan.steps[:state.current_step_index]
            if s.status.value == "complete"
        ]
        if not completed:
            return "Nothing completed yet."
        lines = [f"  {s.step_number}. {s.action} → {str(s.result)[:80]}" for s in completed]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # Interrupt / mid-flight modification
    # ──────────────────────────────────────────────────────────────────────

    async def _handle_interrupt(self, state):
        """
        User sent a message while execution is running.
        LLM receives full execution context + user message.
        If user redirects ("do X instead"), trigger replan.
        """
        from friday.execution.state_manager import ExecutionStatus

        user_message = state.interrupt_message or ""
        state.log_action("Interrupt received", {"message": user_message[:100]})

        # Build context-rich response
        exec_ctx = state.get_context_for_llm()
        messages = [
            {
                "role": "system",
                "content": (
                    self.loop._build_system_prompt() + "\n\n" + exec_ctx
                ),
            },
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self.loop._llm_call(messages)
            # Persist response to session history
            self.loop._persist("user",      user_message)
            self.loop._persist("assistant", response)
            logger.info(f"[Engine] Interrupt handled. Response: {response[:80]}")

            # Check if the user is redirecting execution
            redirect_keywords = [
                "instead", "change to", "do it differently", "use",
                "switch to", "actually", "don't do that", "stop that",
            ]
            if any(kw in user_message.lower() for kw in redirect_keywords):
                # Treat as replan request
                current_step = state.get_current_step()
                if current_step:
                    current_step.error = f"Redirected by user: {user_message[:100]}"
                    await self._attempt_replan(state, current_step)

        except Exception as e:
            logger.error(f"[Engine] Interrupt handler failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _parse_plan_json(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Same extraction logic as loop.py / planner.py."""
        import re
        if "```json" in text:
            try:
                json_str = text.split("```json")[-1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        arr = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if arr:
            try:
                parsed = json.loads(arr.group())
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return None
