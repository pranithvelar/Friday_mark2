import json
import asyncio
import logging
from typing import List, Dict, Any

from friday.agents.friday.session_repair import repair_tool_use_result_pairing

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Handles saving and loading of conversational session history.
    Backed by the 'sessions' SQLite table.
    
    Features (OpenClaw-inspired):
      - Write-lock per session_id to prevent concurrent corruption
      - Automatic transcript repair on load (orphaned tool results, etc.)
    """
    def __init__(self, db_manager):
        self.db = db_manager
        # Keep a reference for summary path compat
        # (loop.py uses session_manager.sessions_dir for summary file path)
        self.sessions_dir = ""
        # Session write-lock registry: one asyncio.Lock per session_id
        self._locks: Dict[str, asyncio.Lock] = {}

    def _conn(self):
        return self.db.get_connection()

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create an asyncio.Lock for the given session_id.
        
        Use as:
            async with session_manager.get_lock(session_id):
                # ... exclusive session access ...
        """
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM sessions WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
        
        messages = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except Exception:
                    pass
            messages.append(msg)

        # --- Transcript repair on load ---
        if messages:
            report = repair_tool_use_result_pairing(messages)
            if report["repaired"]:
                logger.info(
                    f"Session '{session_id}' repaired on load: "
                    f"{report['stats']} — {len(report['repairs'])} fixes applied"
                )
                # Persist the repaired history back to SQLite
                self.overwrite_session(session_id, report["messages"])
                return report["messages"]

        return messages

    def append_message(self, session_id: str, role: str, content: str, tool_calls: Any = None):
        conn = self._conn()
        tc_str = json.dumps(tool_calls) if tool_calls is not None else None
        conn.execute(
            "INSERT INTO sessions (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
            (session_id, role, content, tc_str)
        )
        conn.commit()

    def overwrite_session(self, session_id: str, messages: List[Dict[str, Any]]):
        """Replace the entire session history with repaired messages.
        
        Used after transcript repair to persist the corrected sequence.
        Runs inside a transaction for atomicity.
        """
        conn = self._conn()
        try:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            for msg in messages:
                tc = msg.get("tool_calls")
                tc_str = json.dumps(tc) if tc is not None else None
                conn.execute(
                    "INSERT INTO sessions (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
                    (session_id, msg.get("role", "user"), msg.get("content", ""), tc_str)
                )
            conn.commit()
            logger.info(f"Session '{session_id}' overwritten with {len(messages)} repaired messages")
        except Exception as e:
            logger.error(f"Failed to overwrite session '{session_id}': {e}")

    def clear_session(self, session_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
