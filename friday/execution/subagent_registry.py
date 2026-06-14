"""
Subagent Registry
==================
Tracks spawned sub-executions for the Multi-Agent Planner.
Inspired by OpenClaw's subagent-registry.ts:
  - Depth limits    (MAX_SPAWN_DEPTH)
  - Children limits (MAX_CHILDREN_PER_EXECUTION)
  - Lifecycle: register → running → complete / fail

No external dependencies. Pure in-memory state per process lifetime.
For persistence between restarts, extend with SQLite backing.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ── Safety limits (mirrors OpenClaw defaults) ───────────────────────────────
MAX_SPAWN_DEPTH           = 3   # Max nesting depth of sub-executions
MAX_CHILDREN_PER_EXECUTION = 5  # Max parallel child executions per parent


class SubagentStatus(Enum):
    QUEUED   = "queued"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"


@dataclass
class SubagentRecord:
    execution_id:       str
    parent_exec_id:     Optional[str]
    task_description:   str
    depth:              int
    status:             SubagentStatus = SubagentStatus.QUEUED
    started_at:         Optional[datetime] = None
    ended_at:           Optional[datetime] = None
    child_count:        int = 0
    error:              Optional[str] = None
    result_summary:     Optional[str] = None


class SubagentRegistry:
    """
    In-memory registry of all active and completed sub-executions.

    Enforces:
      - MAX_SPAWN_DEPTH  — prevents recursive explosion
      - MAX_CHILDREN_PER_EXECUTION — prevents too many parallel branches
    """

    def __init__(self):
        self._active:    Dict[str, SubagentRecord] = {}
        self._completed: Dict[str, SubagentRecord] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        task_description: str,
        parent_exec_id: Optional[str] = None,
    ) -> SubagentRecord:
        """
        Register a new sub-execution. Raises ValueError if safety limits hit.
        Returns the SubagentRecord (with a new execution_id).
        """
        depth = self._resolve_depth(parent_exec_id)

        if depth >= MAX_SPAWN_DEPTH:
            raise ValueError(
                f"Cannot spawn sub-execution: max depth {MAX_SPAWN_DEPTH} reached "
                f"(current depth: {depth})."
            )

        if parent_exec_id:
            active_children = self._count_active_children(parent_exec_id)
            if active_children >= MAX_CHILDREN_PER_EXECUTION:
                raise ValueError(
                    f"Cannot spawn sub-execution: parent already has "
                    f"{active_children}/{MAX_CHILDREN_PER_EXECUTION} active children."
                )
            # Increment parent child count
            if parent_exec_id in self._active:
                self._active[parent_exec_id].child_count += 1

        record = SubagentRecord(
            execution_id=str(uuid.uuid4()),
            parent_exec_id=parent_exec_id,
            task_description=task_description,
            depth=depth,
            status=SubagentStatus.QUEUED,
        )
        self._active[record.execution_id] = record
        logger.debug(
            f"[SubagentRegistry] Registered {record.execution_id} "
            f"depth={depth} parent={parent_exec_id}"
        )
        return record

    # ── Status transitions ────────────────────────────────────────────────

    def mark_running(self, execution_id: str):
        rec = self._get_active(execution_id)
        if rec:
            rec.status = SubagentStatus.RUNNING
            rec.started_at = datetime.now()

    def complete(self, execution_id: str, result_summary: str = ""):
        rec = self._active.pop(execution_id, None)
        if rec:
            rec.status = SubagentStatus.COMPLETE
            rec.ended_at = datetime.now()
            rec.result_summary = result_summary
            self._completed[execution_id] = rec
            self._decrement_parent_child(rec.parent_exec_id)

    def fail(self, execution_id: str, error: str):
        rec = self._active.pop(execution_id, None)
        if rec:
            rec.status = SubagentStatus.FAILED
            rec.ended_at = datetime.now()
            rec.error = error
            self._completed[execution_id] = rec
            self._decrement_parent_child(rec.parent_exec_id)

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, execution_id: str) -> Optional[SubagentRecord]:
        return self._active.get(execution_id) or self._completed.get(execution_id)

    def get_active_count_for_parent(self, parent_exec_id: str) -> int:
        return self._count_active_children(parent_exec_id)

    def get_depth(self, execution_id: str) -> int:
        rec = self.get(execution_id)
        return rec.depth if rec else 0

    def active_count(self) -> int:
        return len(self._active)

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_active(self, execution_id: str) -> Optional[SubagentRecord]:
        return self._active.get(execution_id)

    def _resolve_depth(self, parent_exec_id: Optional[str]) -> int:
        if not parent_exec_id:
            return 0
        parent = self._active.get(parent_exec_id) or self._completed.get(parent_exec_id)
        return (parent.depth + 1) if parent else 1

    def _count_active_children(self, parent_exec_id: str) -> int:
        return sum(
            1 for r in self._active.values()
            if r.parent_exec_id == parent_exec_id
            and r.status in (SubagentStatus.QUEUED, SubagentStatus.RUNNING)
        )

    def _decrement_parent_child(self, parent_exec_id: Optional[str]):
        if parent_exec_id and parent_exec_id in self._active:
            self._active[parent_exec_id].child_count = max(
                0, self._active[parent_exec_id].child_count - 1
            )
