"""
Execution State Manager
========================
Tracks everything about an in-flight execution.

Key concepts (OpenClaw-inspired):
  - ExecutionStep  : one atomic action in the plan
  - ExecutionPlan  : ordered list of steps + metadata
  - ExecutionState : live mutable state of a running plan
  - ExecutionStateManager : global registry of all active/completed executions

The ExecutionState.get_context_for_llm() output is injected into Friday's
context on EVERY LLM call while an execution is running — this is how
Friday stays aware of what the engine is doing in real time.
"""

import uuid
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    PENDING    = "pending"
    PLANNING   = "planning"
    RUNNING    = "running"
    WAITING    = "waiting"
    ERROR      = "error"
    RETRYING   = "retrying"
    REPLANNING = "replanning"
    PAUSED     = "paused"
    COMPLETE   = "complete"
    FAILED     = "failed"


@dataclass
class ExecutionStep:
    """One atomic action in the execution plan."""
    step_id:            str
    step_number:        int
    action:             str
    tool_category:      str           # "browser" | "search" | "bash" | "mcp" | "memory" | "none"
    reasoning:          str
    estimated_seconds:  int = 10
    status:             ExecutionStatus = ExecutionStatus.PENDING
    result:             Optional[Any] = None
    error:              Optional[str] = None
    started_at:         Optional[datetime] = None
    completed_at:       Optional[datetime] = None
    retry_count:        int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id":          self.step_id,
            "step_number":      self.step_number,
            "action":           self.action,
            "tool_category":    self.tool_category,
            "reasoning":        self.reasoning,
            "estimated_seconds":self.estimated_seconds,
            "status":           self.status.value,
            "result":           str(self.result)[:200] if self.result else None,
            "error":            self.error,
            "started_at":       self.started_at.isoformat() if self.started_at else None,
            "completed_at":     self.completed_at.isoformat() if self.completed_at else None,
            "retry_count":      self.retry_count,
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan produced by the MultiAgentPlanner."""
    plan_id:                    str
    query:                      str
    steps:                      List[ExecutionStep]
    estimated_duration_seconds: int
    complexity:                 str
    created_at:                 datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id":                   self.plan_id,
            "query":                     self.query,
            "steps":                     [s.to_dict() for s in self.steps],
            "estimated_duration_seconds":self.estimated_duration_seconds,
            "complexity":                self.complexity,
            "created_at":                self.created_at.isoformat(),
        }


class ExecutionState:
    """
    Mutable live state of a running execution.

    Thread/async safety: all mutations happen inside the engine's
    async loop — no locking needed for single-coroutine access.
    """

    def __init__(self, execution_id: str, plan: ExecutionPlan, session_id: str):
        self.execution_id        = execution_id
        self.plan                = plan
        self.session_id          = session_id

        # ── Live state ────────────────────────────────────────────────────
        self.status              = ExecutionStatus.PENDING
        self.current_step_index  = 0
        self.progress_percent    = 0.0
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # ── Execution log (rolling, max 100 entries) ──────────────────────
        # This is injected into LLM context so Friday stays aware.
        self.execution_log: List[Dict[str, Any]] = []

        # ── Error tracking ────────────────────────────────────────────────
        self.errors: List[Dict[str, Any]] = []

        # ── Learning data collected during this execution ─────────────────
        self.learned_patterns: List[Dict[str, Any]] = []

        # ── Interrupt support (mid-flight modification) ───────────────────
        self.interrupt_requested  = False
        self.interrupt_message: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start(self):
        self.status     = ExecutionStatus.RUNNING
        self.started_at = datetime.now()
        self._log_internal("execution_started", {
            "plan_id":     self.plan.plan_id,
            "total_steps": len(self.plan.steps),
        })

    def get_current_step(self) -> Optional[ExecutionStep]:
        if self.current_step_index < len(self.plan.steps):
            return self.plan.steps[self.current_step_index]
        return None

    def advance_step(self):
        self.current_step_index += 1
        self._recalculate_progress()

    def update_step_status(
        self,
        step_id: str,
        status: ExecutionStatus,
        result: Any = None,
        error: Optional[str] = None,
    ):
        for step in self.plan.steps:
            if step.step_id != step_id:
                continue
            step.status = status
            step.result = result
            step.error  = error
            if status == ExecutionStatus.RUNNING:
                step.started_at = datetime.now()
            elif status in (ExecutionStatus.COMPLETE, ExecutionStatus.FAILED):
                step.completed_at = datetime.now()
            break

    # ──────────────────────────────────────────────────────────────────────
    # Logging & context injection
    # ──────────────────────────────────────────────────────────────────────

    def log_action(self, action: str, details: Dict[str, Any]):
        """
        Log a real-time action. The last N entries are injected into
        the LLM system context via get_context_for_llm().
        """
        entry = {
            "timestamp":   datetime.now().isoformat(),
            "action":      action,
            "details":     details,
            "step_number": self.current_step_index + 1,
        }
        self.execution_log.append(entry)
        # Rolling window — never exceed 100 entries
        if len(self.execution_log) > 100:
            self.execution_log = self.execution_log[-100:]

    def get_context_for_llm(self) -> str:
        """
        Format execution state as a string block to inject into Friday's
        system prompt while an execution is running.

        Example output:
          [EXECUTION STATUS]
          Task: Research about quantum computing
          Progress: 43%
          Step: 3/7
          Current Action: Searching Google for "quantum computing 2024"
          Tool: browser

          [RECENT ACTIONS]
          • Opened browser: {"url": "google.com"}
          • Typed query: {"text": "quantum computing 2024"}
        """
        current_step = self.get_current_step()
        parts = [
            "[EXECUTION STATUS]",
            f"Task: {self.plan.query}",
            f"Progress: {self.progress_percent:.0f}%",
            f"Step: {self.current_step_index + 1}/{len(self.plan.steps)}",
            f"Engine Status: {self.status.value}",
        ]
        if current_step:
            parts += [
                f"Current Action: {current_step.action}",
                f"Tool: {current_step.tool_category}",
            ]

        if self.execution_log:
            parts.append("\n[RECENT ACTIONS]")
            for entry in self.execution_log[-5:]:
                detail_str = json.dumps(entry["details"])[:120]
                parts.append(f"• {entry['action']}: {detail_str}")

        if self.errors:
            last_err = self.errors[-1]
            parts.append(f"\n[LAST ERROR] {last_err.get('error', 'unknown')}")

        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # Progress & status queries
    # ──────────────────────────────────────────────────────────────────────

    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Returns a structured summary for "what's the progress?" queries.
        SmartRouter returns this instantly without any LLM call.
        """
        current_step = self.get_current_step()
        return {
            "status":          self.status.value,
            "progress_percent":self.progress_percent,
            "current_step":    self.current_step_index + 1,
            "total_steps":     len(self.plan.steps),
            "current_action":  current_step.action if current_step else "Complete",
            "current_tool":    current_step.tool_category if current_step else None,
            "task":            self.plan.query,
            "last_log":        self.execution_log[-1] if self.execution_log else None,
        }

    def format_progress_response(self) -> str:
        """Human-readable progress response for Friday to return directly."""
        s = self.get_progress_summary()
        name_prefix = "Sir"  # Personalization hook — override in SmartRouter if needed

        if s["status"] == "complete":
            return f"The task is complete, {name_prefix}. All {s['total_steps']} steps finished successfully."

        if s["status"] == "failed":
            return f"I encountered an issue, {name_prefix}. The execution has stopped. Last error logged."

        return (
            f"Currently at step {s['current_step']} of {s['total_steps']} "
            f"({s['progress_percent']:.0f}% complete), {name_prefix}.\n"
            f"Action: {s['current_action']}\n"
            f"Tool in use: {s['current_tool'] or 'none'}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Error recording
    # ──────────────────────────────────────────────────────────────────────

    def record_error(self, error: str, step_id: str, context: Dict[str, Any]):
        self.errors.append({
            "timestamp":   datetime.now().isoformat(),
            "error":       error,
            "step_id":     step_id,
            "context":     context,
        })

    # ──────────────────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────────────────

    def learn_pattern(self, pattern_type: str, data: Dict[str, Any]):
        self.learned_patterns.append({
            "timestamp":    datetime.now().isoformat(),
            "pattern_type": pattern_type,
            "data":         data,
            "query":        self.plan.query,
        })

    # ──────────────────────────────────────────────────────────────────────
    # Interrupt (mid-flight modification)
    # ──────────────────────────────────────────────────────────────────────

    def request_interrupt(self, message: str):
        self.interrupt_requested = True
        self.interrupt_message   = message

    def clear_interrupt(self):
        self.interrupt_requested = False
        self.interrupt_message   = None

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _recalculate_progress(self):
        total = len(self.plan.steps)
        if total == 0:
            self.progress_percent = 100.0
            return
        completed = sum(
            1 for s in self.plan.steps
            if s.status == ExecutionStatus.COMPLETE
        )
        self.progress_percent = (completed / total) * 100.0

    def _log_internal(self, event: str, data: Dict[str, Any]):
        logger.debug(f"[ExecutionState:{self.execution_id[:8]}] {event}: {data}")


# ── Global state manager ──────────────────────────────────────────────────────

class ExecutionStateManager:
    """
    Global in-process registry of all active and completed executions.
    One instance is created in terminal_chat.py and shared across all components.
    """

    def __init__(self):
        self.active_executions:    Dict[str, ExecutionState] = {}
        self.completed_executions: Dict[str, ExecutionState] = {}

    def create_execution(self, plan: ExecutionPlan, session_id: str) -> ExecutionState:
        execution_id = str(uuid.uuid4())
        state = ExecutionState(execution_id, plan, session_id)
        self.active_executions[execution_id] = state
        logger.info(f"[StateManager] Created execution {execution_id[:8]} for session {session_id}")
        return state

    def get_execution(self, execution_id: str) -> Optional[ExecutionState]:
        return self.active_executions.get(execution_id) or \
               self.completed_executions.get(execution_id)

    def get_session_execution(self, session_id: str) -> Optional[ExecutionState]:
        """Get the currently active execution for a session."""
        for state in self.active_executions.values():
            if state.session_id == session_id:
                return state
        return None

    def has_active_execution(self, session_id: str) -> bool:
        return self.get_session_execution(session_id) is not None

    def complete_execution(self, execution_id: str):
        state = self.active_executions.pop(execution_id, None)
        if state:
            state.status       = ExecutionStatus.COMPLETE
            state.completed_at = datetime.now()
            self.completed_executions[execution_id] = state
            logger.info(f"[StateManager] Execution {execution_id[:8]} completed")

    def fail_execution(self, execution_id: str, reason: str):
        state = self.active_executions.pop(execution_id, None)
        if state:
            state.status       = ExecutionStatus.FAILED
            state.completed_at = datetime.now()
            state.record_error(reason, "engine", {})
            self.completed_executions[execution_id] = state
            logger.warning(f"[StateManager] Execution {execution_id[:8]} failed: {reason}")

    def active_count(self) -> int:
        return len(self.active_executions)
