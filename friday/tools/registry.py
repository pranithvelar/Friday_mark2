"""
Tool Registry
=============
Auto-discovers all tools and builds their schemas.
Provides tool dispatch for both medium and complex handlers.

Usage:
    registry = ToolRegistry()
    registry.discover()                         # scan all tools/ subdirs
    schemas = registry.get_schema_general()     # for medium handler
    schemas = registry.get_schema_for("friday") # for friday agent
    result  = await registry.execute("web_search", query="AI trends")

Adding a new tool:
    1. Create tools/<category>/<name>.py
    2. Implement BaseTool (name, description, scope, parameters, run())
    3. Done. Registry auto-discovers it.
"""

import importlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Discovers and manages all tools.
    One instance per process — created at startup in chat.py.
    """

    def __init__(self):
        self._tools = {}  # tool_name → tool instance

    def discover(self) -> int:
        """
        Scan all subdirectories of tools/ and register BaseTool subclasses.
        Returns the number of tools discovered.
        """
        tools_path = Path(__file__).parent
        count = 0
        skip_files = {"__init__.py", "base_tool.py", "registry.py", "tools_legacy.py"}

        for py_file in tools_path.rglob("*.py"):
            if py_file.name in skip_files or py_file.name.startswith("_"):
                continue
            # Build module path: tools/general/web_search.py → friday.tools.general.web_search
            rel = py_file.relative_to(tools_path.parent)
            module_name = "friday." + str(rel).replace("\\", ".").replace("/", ".")[:-3]
            try:
                module = importlib.import_module(module_name)
                from friday.tools.base_tool import BaseTool
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    try:
                        if (isinstance(attr, type)
                                and issubclass(attr, BaseTool)
                                and attr is not BaseTool
                                and attr.name):
                            instance = attr()
                            self._tools[instance.name] = instance
                            logger.info(f"[ToolRegistry] Registered tool: {instance.name} (scope={instance.scope})")
                            count += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[ToolRegistry] Could not load {py_file.name}: {e}")

        return count

    def get_schema_general(self) -> List[Dict[str, Any]]:
        """
        Returns schemas for tools with scope='general'.
        Used by MediumHandler — safe for 1-3 tool calls.
        """
        return [
            t.to_schema() for t in self._tools.values()
            if t.scope == "general"
        ]

    def get_schema_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Returns schemas for general tools + agent-specific tools.
        Used by ComplexHandler and individual agents.
        """
        return [
            t.to_schema() for t in self._tools.values()
            if t.scope in ("general", f"agent:{agent_id}")
        ]

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """All tools — for the main Friday agent."""
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Dispatch a tool call by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found", "available": list(self._tools.keys())}
        try:
            return await tool.run(**kwargs)
        except Exception as e:
            logger.error(f"[ToolRegistry] Tool '{tool_name}' failed: {e}")
            return {"error": str(e), "tool": tool_name}

    def get(self, tool_name: str):
        return self._tools.get(tool_name)

    def list_all(self) -> List[str]:
        return list(self._tools.keys())

    def summary(self) -> str:
        if not self._tools:
            return "No tools registered."
        lines = [f"Registered tools ({len(self._tools)}):"]
        for t in self._tools.values():
            lines.append(f"  • {t.name} [{t.scope}]: {t.description[:60]}")
        return "\n".join(lines)
