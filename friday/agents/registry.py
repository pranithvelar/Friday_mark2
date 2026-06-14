"""
Agent Registry
==============
Auto-discovers all specialized agents at startup.
The ComplexHandler queries this registry to find which
agents can handle a given task.

Usage:
    registry = AgentRegistry()
    registry.discover()                          # scan agents/specialized/
    agents = registry.find_for_task("research")  # returns matching agents
    all_agents = registry.list_all()             # returns all registered agents

Adding a new agent:
    1. Create agents/specialized/<name>/agent.py
    2. Implement BaseAgent (agent_id, description, capabilities, run())
    3. Done. Registry auto-discovers it on next startup.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Discovers and manages all specialized agents.
    One instance per process — created at startup in chat.py.
    """

    def __init__(self):
        self._agents = {}  # agent_id → agent instance

    def discover(self) -> int:
        """
        Scan agents/specialized/ and register all BaseAgent subclasses found.
        Returns the number of agents discovered.
        """
        specialized_path = Path(__file__).parent / "specialized"
        if not specialized_path.exists():
            return 0

        count = 0
        for subdir in specialized_path.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            agent_module_path = subdir / "agent.py"
            if not agent_module_path.exists():
                continue
            try:
                module_name = f"friday.agents.specialized.{subdir.name}.agent"
                module = importlib.import_module(module_name)
                # Find subclasses of BaseAgent in this module
                from friday.agents.base_agent import BaseAgent
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    try:
                        if (isinstance(attr, type)
                                and issubclass(attr, BaseAgent)
                                and attr is not BaseAgent
                                and attr.agent_id):
                            instance = attr()
                            self._agents[instance.agent_id] = instance
                            logger.info(f"[AgentRegistry] Registered agent: {instance.agent_id}")
                            count += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[AgentRegistry] Could not load {subdir.name}: {e}")

        return count

    def find_for_task(self, task_description: str) -> List:
        """
        Return agents whose capabilities match keywords in the task description.
        Used by ComplexHandler to select which agents to deploy.
        """
        task_lower = task_description.lower()
        matched = []
        for agent in self._agents.values():
            for cap in (agent.capabilities or []):
                if cap.lower() in task_lower:
                    matched.append(agent)
                    break
        return matched

    def get(self, agent_id: str):
        """Get a specific agent by ID."""
        return self._agents.get(agent_id)

    def list_all(self) -> List:
        """Return all registered agents."""
        return list(self._agents.values())

    def summary(self) -> str:
        """Human-readable summary for awareness/status_reporter."""
        if not self._agents:
            return "No specialized agents registered."
        lines = [f"Registered agents ({len(self._agents)}):"]
        for agent in self._agents.values():
            caps = ", ".join(agent.capabilities[:3]) if agent.capabilities else "none"
            lines.append(f"  • {agent.agent_id}: {agent.description} [{caps}]")
        return "\n".join(lines)
