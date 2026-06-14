"""
Parallel Executor
=================
Runs multiple agents concurrently using asyncio.gather().

Used by the ComplexHandler when a plan has steps that can run simultaneously.

Example:
    executor = ParallelExecutor(agent_registry)
    results = await executor.run([
        ("researcher", "find AI trends for 2025"),
        ("whatsapp", "read last 10 messages from Mum"),
    ])
    # results[0] = researcher output, results[1] = whatsapp output

Safety:
    - Per-agent timeout (default 60s)
    - Each agent failure returns an error dict, does not kill other agents
    - Max concurrent agents = MAX_PARALLEL (default 5, mirrors subagent_registry limit)
"""

import asyncio
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

MAX_PARALLEL      = 5    # mirrors SubagentRegistry.MAX_CHILDREN_PER_EXECUTION
AGENT_TIMEOUT_SEC = 60


class ParallelExecutor:
    """
    Wraps asyncio.gather() for safe multi-agent parallel execution.
    """

    def __init__(self, agent_registry):
        self.registry = agent_registry

    async def run(
        self,
        tasks: List[Tuple[str, str]],   # [(agent_id, task_description), ...]
        context: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute all tasks in parallel.
        Returns a list of results in the same order as tasks.

        Args:
            tasks:   list of (agent_id, task_description) tuples
            context: shared context dict passed to each agent
        """
        if not tasks:
            return []

        if len(tasks) > MAX_PARALLEL:
            logger.warning(
                f"[ParallelExecutor] {len(tasks)} tasks requested but max is {MAX_PARALLEL}. "
                f"Truncating to first {MAX_PARALLEL}."
            )
            tasks = tasks[:MAX_PARALLEL]

        context = context or {}
        coroutines = [
            self._run_one(agent_id, task, context)
            for agent_id, task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Normalize exceptions into error dicts
        normalized = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_id = tasks[i][0]
                logger.error(f"[ParallelExecutor] Agent '{agent_id}' failed: {result}")
                normalized.append({"error": str(result), "agent": agent_id})
            else:
                normalized.append(result)

        return normalized

    async def _run_one(
        self,
        agent_id: str,
        task: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single agent with timeout."""
        agent = self.registry.get(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found", "agent": agent_id}

        try:
            result = await asyncio.wait_for(
                agent.run(task=task, context=context),
                timeout=AGENT_TIMEOUT_SEC,
            )
            logger.info(f"[ParallelExecutor] Agent '{agent_id}' completed task")
            return result
        except asyncio.TimeoutError:
            return {"error": f"Agent '{agent_id}' timed out after {AGENT_TIMEOUT_SEC}s", "agent": agent_id}
        except Exception as e:
            return {"error": str(e), "agent": agent_id}
