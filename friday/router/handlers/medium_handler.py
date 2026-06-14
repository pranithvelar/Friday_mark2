"""
Tier 2B — Medium Handler
=========================
Delegates to the EXISTING AgentLoop with a hard max_steps=3 cap.
No modifications to loop.py are made — the cap is passed as an argument
to loop.run(), which already supports it.

Target latency: <2 seconds for typical single-tool operations.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MediumHandler:
    """
    Wraps the existing AgentLoop instance with a 3-tool cap.
    The same loop object used by terminal_chat.py is passed here —
    no duplication, no new model instantiation.
    """

    MAX_TOOLS = 3

    def __init__(self, agent_loop):
        """
        Parameters
        ----------
        agent_loop : AgentLoop
            The fully initialised AgentLoop from terminal_chat.py.
            It already has tools registered, session loaded, etc.
        """
        self.loop = agent_loop

    async def handle(self, query: str, session_id: str, bundle=None) -> Dict[str, Any]:
        """
        Run the AgentLoop with max_steps=3 (= max 3 tool invocations).
        If a pre-assembled ContextBundle is provided, the query is pre-augmented
        with memory/profile context before being handed to AgentLoop.

        Returns
        -------
        dict with keys:
            text        : str  — final response text
            tools_used  : list — list of tool names that were called
        """
        # Snapshot history length before to diff tool calls used
        history_before = len(self.loop._history)

        # If SmartRouter already assembled a context bundle, use it to
        # pre-augment the message — AgentLoop will still do its own _pre_search
        # which will now also find conversation chunks we've been indexing.
        message = query
        if bundle is not None and not bundle.is_empty:
            message = bundle.augment(query)

        response = await self.loop.run(message, max_steps=self.MAX_TOOLS)

        # Detect which tools were actually called by inspecting new history entries
        tools_used = self._extract_tools_from_history(history_before)

        return {
            "text": response,
            "tools_used": tools_used,
        }

    def _extract_tools_from_history(self, history_start: int) -> List[str]:
        """
        Scan new history entries (added after history_start) for assistant
        tool-call messages to build the list of tools actually used.
        The loop stores tool calls in the history as assistant messages
        containing ```json ... ``` blocks.
        """
        import json
        import re
        tools_used = []
        new_entries = self.loop._history[history_start:]
        for msg in new_entries:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if "```json" not in content:
                continue
            try:
                json_str = content.split("```json")[-1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_name = parsed["name"]
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
            except Exception:
                pass
        return tools_used
