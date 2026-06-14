"""
Session Transcript Repair — Battle-hardened transcript healing.

Ported from OpenClaw's session-transcript-repair.ts concept.
Handles:
  - Orphaned tool_results (tool result without a preceding tool call)
  - Orphaned tool_calls (tool call without a following result)
  - Corrupted / malformed JSON in tool-call content
  - Role sequence violations (consecutive same-role messages)

Usage:
    from friday.agents.friday.session_repair import repair_tool_use_result_pairing
    report = repair_tool_use_result_pairing(messages)
    repaired_messages = report["messages"]
"""

import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for detecting tool-call JSON blocks in assistant messages
# ---------------------------------------------------------------------------
_TOOL_CALL_PATTERN = re.compile(
    r'```json\s*\n?\s*\{[^}]*"name"\s*:\s*"[^"]+?"[^}]*\}\s*\n?\s*```',
    re.DOTALL,
)

_BARE_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+?"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}',
    re.DOTALL,
)

# Matches the "Result: ..." prefix used by the agent loop for tool outputs
_TOOL_RESULT_PREFIX = re.compile(r'^Result:\s', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Identifier preservation regex — used by compaction prompts
# ---------------------------------------------------------------------------
IDENTIFIER_PATTERN = re.compile(
    r'(?:'
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'  # UUID
    r'|[0-9a-fA-F]{40}'   # SHA-1
    r'|[0-9a-fA-F]{64}'   # SHA-256
    r'|\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b'  # IPv4 + optional port
    r'|https?://[^\s<>"\')]+' # URLs
    r'|/[\w./-]{3,}'      # Unix file paths
    r'|[A-Z]:\\[\w.\\/-]{3,}'  # Windows file paths
    r')',
    re.MULTILINE,
)


def _message_has_tool_call(msg: Dict[str, Any]) -> bool:
    """Check if an assistant message contains a tool-call JSON block."""
    content = msg.get("content", "")
    if not content:
        return False
    return bool(_TOOL_CALL_PATTERN.search(content) or _BARE_TOOL_CALL_PATTERN.search(content))


def _message_is_tool_result(msg: Dict[str, Any]) -> bool:
    """Check if a user message is actually a tool result."""
    content = msg.get("content", "")
    if not content:
        return False
    return bool(_TOOL_RESULT_PREFIX.match(content))


def _try_fix_json_in_content(content: str) -> str:
    """Attempt to repair malformed JSON blocks in assistant messages.
    
    Handles common issues:
      - Unquoted keys: {name: "x"} -> {"name": "x"}
      - Trailing commas: {"a": 1,} -> {"a": 1}
      - Single quotes: {'name': 'x'} -> {"name": "x"}
    """
    # Broad pattern to extract ```json ... ``` blocks (handles nested braces)
    _BROAD_JSON_BLOCK = re.compile(r'```json\s*\n(.*?)\n\s*```', re.DOTALL)

    def fix_block(text: str) -> str:
        # Fix unquoted keys
        text = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1 "\2":', text)
        # Fix single quotes around values
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
        # Fix trailing commas
        text = re.sub(r',\s*([\]}])', r'\1', text)
        return text

    def replace_json_block(m):
        inner = m.group(1).strip()
        fixed = fix_block(inner)
        # Verify it actually parses now
        try:
            json.loads(fixed)
            return f'```json\n{fixed}\n```'
        except (ValueError, json.JSONDecodeError):
            return m.group(0)  # return original if still broken

    result = _BROAD_JSON_BLOCK.sub(replace_json_block, content)
    return result


def repair_tool_use_result_pairing(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Repair a transcript so tool calls and tool results are properly paired.
    
    Returns:
        {
            "messages": List[Dict] — the repaired message list,
            "repairs": List[str] — human-readable descriptions of each repair,
            "repaired": bool — True if any repairs were made,
            "stats": {
                "orphaned_results_dropped": int,
                "orphaned_calls_patched": int,
                "json_blocks_repaired": int,
                "consecutive_roles_merged": int,
            }
        }
    """
    if not messages:
        return {
            "messages": [],
            "repairs": [],
            "repaired": False,
            "stats": {
                "orphaned_results_dropped": 0,
                "orphaned_calls_patched": 0,
                "json_blocks_repaired": 0,
                "consecutive_roles_merged": 0,
            },
        }

    repairs: List[str] = []
    stats = {
        "orphaned_results_dropped": 0,
        "orphaned_calls_patched": 0,
        "json_blocks_repaired": 0,
        "consecutive_roles_merged": 0,
    }

    # ------------------------------------------------------------------
    # Pass 1: Repair malformed JSON in tool-call blocks
    # ------------------------------------------------------------------
    working = []
    for msg in messages:
        msg_copy = dict(msg)
        if msg_copy.get("role") == "assistant":
            original = msg_copy.get("content", "")
            fixed = _try_fix_json_in_content(original)
            if fixed != original:
                msg_copy["content"] = fixed
                stats["json_blocks_repaired"] += 1
                repairs.append(
                    f"Repaired malformed JSON in assistant message: "
                    f"{original[:60]}..."
                )
        working.append(msg_copy)

    # ------------------------------------------------------------------
    # Pass 2: Fix orphaned tool results and orphaned tool calls
    # ------------------------------------------------------------------
    repaired = []
    i = 0
    while i < len(working):
        msg = working[i]
        role = msg.get("role", "")

        # Case A: Tool result without a preceding tool call
        if role == "user" and _message_is_tool_result(msg):
            # Look back: was the previous message an assistant with a tool call?
            if repaired and repaired[-1].get("role") == "assistant" and _message_has_tool_call(repaired[-1]):
                # Properly paired — keep it
                repaired.append(msg)
            else:
                # Orphaned tool result — drop it
                stats["orphaned_results_dropped"] += 1
                content_preview = msg.get("content", "")[:80]
                repairs.append(
                    f"Dropped orphaned tool result (no preceding tool call): "
                    f"{content_preview}..."
                )
                i += 1
                continue

        # Case B: Assistant message with a tool call — check for following result
        elif role == "assistant" and _message_has_tool_call(msg):
            repaired.append(msg)
            # Peek ahead: is there a tool result next?
            if i + 1 < len(working):
                next_msg = working[i + 1]
                if next_msg.get("role") == "user" and _message_is_tool_result(next_msg):
                    # Properly paired — the next iteration will handle it
                    pass
                else:
                    # Orphaned tool call — inject a synthetic failure result
                    stats["orphaned_calls_patched"] += 1
                    repairs.append(
                        f"Injected synthetic tool failure for orphaned tool call: "
                        f"{msg.get('content', '')[:60]}..."
                    )
                    repaired.append({
                        "role": "user",
                        "content": "Result: Tool execution was interrupted (session crash recovery). "
                                   "The tool did not return a result."
                    })
            else:
                # Tool call at the very end — inject failure result
                stats["orphaned_calls_patched"] += 1
                repairs.append(
                    f"Injected synthetic tool failure for trailing tool call: "
                    f"{msg.get('content', '')[:60]}..."
                )
                repaired.append({
                    "role": "user",
                    "content": "Result: Tool execution was interrupted (session crash recovery). "
                               "The tool did not return a result."
                })
        else:
            repaired.append(msg)

        i += 1

    # ------------------------------------------------------------------
    # Pass 3: Merge consecutive same-role messages (except system)
    # ------------------------------------------------------------------
    merged = []
    for msg in repaired:
        if (
            merged
            and msg.get("role") == merged[-1].get("role")
            and msg.get("role") != "system"
        ):
            # Merge into previous
            merged[-1]["content"] = (
                merged[-1].get("content", "") + "\n" + msg.get("content", "")
            )
            stats["consecutive_roles_merged"] += 1
            repairs.append(
                f"Merged consecutive {msg['role']} messages"
            )
        else:
            merged.append(msg)

    is_repaired = any(v > 0 for v in stats.values())

    if is_repaired:
        logger.info(
            f"Transcript repair complete: {stats} | "
            f"{len(repairs)} repairs applied to {len(messages)} messages"
        )

    return {
        "messages": merged,
        "repairs": repairs,
        "repaired": is_repaired,
        "stats": stats,
    }


def extract_identifiers(text: str) -> List[str]:
    """Extract all opaque identifiers from text for preservation checking."""
    return IDENTIFIER_PATTERN.findall(text)
