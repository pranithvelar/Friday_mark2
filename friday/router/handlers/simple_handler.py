"""
Tier 2A — Simple Handler
=========================
Handles SIMPLE queries entirely with direct SQLite + in-memory lookup.
NO LLM call. Target latency: <200ms.

Connected to REAL project tables:
  - facts          (calendar / temporal events)
  - user_profile   (via UserPersonalization)
"""

import logging
import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class SimpleHandler:
    """
    Handles SIMPLE tier queries without any LLM call.
    Injected with the real db_manager and personalization objects from terminal_chat.py.
    """

    def __init__(self, db_manager, personalization, searcher=None):
        self.db = db_manager
        self.personalization = personalization
        self.searcher = searcher  # optional: used for context-enhanced greetings

    # ──────────────────────────────────────────────────────────────
    # Public dispatch
    # ──────────────────────────────────────────────────────────────

    async def handle(self, query: str, category, session_id: str) -> str:
        """Route to the correct simple handler based on category."""
        from friday.router.intent_classifier import QueryCategory
        dispatch = {
            QueryCategory.GREETING:       self._handle_greeting,
            QueryCategory.CALENDAR_QUERY: self._handle_calendar_query,
            QueryCategory.FACT_RECALL:    self._handle_fact_recall,
            QueryCategory.TEMPORAL_FACT:  self._handle_temporal_fact,
            QueryCategory.PROGRESS_QUERY: self._handle_progress_query,
            QueryCategory.GENERAL_CHAT:   self._handle_general_chat,
        }
        handler = dispatch.get(category)
        if handler:
            return await handler(query, session_id)
        # Unknown simple category — surface gracefully
        return await self._handle_fact_recall(query, session_id)

    # ──────────────────────────────────────────────────────────────
    # GENERAL CHAT — instant acknowledgement, no DB, no LLM
    # ──────────────────────────────────────────────────────────────

    async def _handle_general_chat(self, query: str, session_id: str) -> str:
        """Short casual inputs — direct JARVIS-style acknowledgement, zero latency."""
        q = query.lower().strip()
        # Cancellation / clearing intent
        if any(w in q for w in ["cancel", "forget it", "never mind", "clear", "drop", "skip"]):
            return "Understood, Sir. Consider it done."
        # Affirmation
        if any(w in q for w in ["ok", "okay", "sure", "got it", "noted", "alright", "fine", "yes", "yep"]):
            return "Noted, Sir."
        # Gratitude
        if any(w in q for w in ["thanks", "thank you", "cheers", "appreciate"]):
            return "Always, Sir."
        # Generic short input — stay concise
        return "Understood. What would you like me to do?"

    # ──────────────────────────────────────────────────────────────
    # PROGRESS QUERY — handled via execution state manager
    # (SmartRouter intercepts this before reaching here, but kept as fallback)
    # ──────────────────────────────────────────────────────────────

    async def _handle_progress_query(self, query: str, session_id: str) -> str:
        return "I am ready and waiting for your next instruction, Sir."

    # ──────────────────────────────────────────────────────────────
    # GREETING — instant, no DB
    # ──────────────────────────────────────────────────────────────

    async def _handle_greeting(self, query: str, session_id: str) -> str:
        name = "Sir"
        if self.personalization:
            stored = self.personalization.get_preference("address_as")
            # Only use the stored value if it looks like a real name/title
            # Reject single words that are common response words, not names
            _JUNK_VALUES = {"done", "ok", "yes", "no", "sure", "okay", "thanks",
                            "correct", "right", "great", "good", "fine", "alright"}
            if stored and stored.lower().strip() not in _JUNK_VALUES and len(stored.strip()) >= 2:
                name = stored.strip().title()
        greetings = [
            f"Good day, {name}. How may I assist you?",
            f"Hello, {name}. What can I do for you?",
            f"At your service, {name}.",
        ]
        return greetings[hash(query.lower().strip()) % len(greetings)]


    # ──────────────────────────────────────────────────────────────
    # CALENDAR QUERY — direct query on facts table
    # ──────────────────────────────────────────────────────────────

    async def _handle_calendar_query(self, query: str, session_id: str) -> str:
        time_range = self._extract_time_range(query)
        try:
            conn = self.db.get_connection()
            rows = conn.execute(
                """
                SELECT content, date_start, date_end, importance
                FROM facts
                WHERE status = 'active'
                  AND date_end >= ?
                  AND date_start <= ?
                ORDER BY date_start ASC
                """,
                (time_range["start"].isoformat(), time_range["end"].isoformat())
            ).fetchall()
        except Exception as e:
            logger.warning(f"Calendar query failed: {e}")
            return "I could not retrieve your calendar events right now."

        if not rows:
            return f"You have no events scheduled {time_range['description']}."

        lines = [f"You have {len(rows)} event(s) {time_range['description']}:\n"]
        for row in rows:
            try:
                ds = datetime.datetime.fromisoformat(row["date_start"])
                date_str = ds.strftime("%A, %B %d at %I:%M %p")
                lines.append(f"  • {date_str}: {row['content']}")
            except Exception:
                lines.append(f"  • {row['content']}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # FACT RECALL — direct query on user_profile table
    # ──────────────────────────────────────────────────────────────

    async def _handle_fact_recall(self, query: str, session_id: str) -> str:
        if not self.personalization:
            return "I have no profile data stored yet."

        # Try to match a specific key
        fact_key = self._extract_fact_key(query)
        if fact_key:
            val = self.personalization.get_fact(fact_key) or \
                  self.personalization.get_preference(fact_key)
            if val:
                return f"Your {fact_key.replace('_', ' ')} is: {val}."

        # Fallback: dump full profile context
        ctx = self.personalization.get_context_string()
        if ctx:
            return ctx
        return "I do not have that information stored yet, Sir."

    # ──────────────────────────────────────────────────────────────
    # TEMPORAL FACT — query facts for a specific named event
    # ──────────────────────────────────────────────────────────────

    async def _handle_temporal_fact(self, query: str, session_id: str) -> str:
        now = datetime.datetime.now()
        try:
            conn = self.db.get_connection()
            rows = conn.execute(
                """
                SELECT content, date_start, date_end
                FROM facts
                WHERE status = 'active'
                  AND date_end >= ?
                ORDER BY date_start ASC
                LIMIT 5
                """,
                (now.isoformat(),)
            ).fetchall()
        except Exception as e:
            logger.warning(f"Temporal fact query failed: {e}")
            return "I could not retrieve that information right now."

        if not rows:
            return "No upcoming events or deadlines found in my records."

        lines = ["Upcoming events from your records:\n"]
        for row in rows:
            try:
                ds = datetime.datetime.fromisoformat(row["date_start"])
                days_until = (ds.date() - now.date()).days
                if days_until == 0:
                    label = "TODAY"
                elif days_until == 1:
                    label = "TOMORROW"
                elif days_until < 0:
                    label = f"{abs(days_until)} days ago"
                else:
                    label = f"in {days_until} days"
                lines.append(f"  • {row['content']} — {label}")
            except Exception:
                lines.append(f"  • {row['content']}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _extract_time_range(self, query: str) -> Dict[str, Any]:
        """Parse natural-language time expressions from query."""
        q = query.lower()
        now = datetime.datetime.now()

        if "today" in q:
            return {
                "start": now.replace(hour=0, minute=0, second=0, microsecond=0),
                "end":   now.replace(hour=23, minute=59, second=59, microsecond=999999),
                "description": "today",
            }
        if "tomorrow" in q:
            tom = now + datetime.timedelta(days=1)
            return {
                "start": tom.replace(hour=0, minute=0, second=0, microsecond=0),
                "end":   tom.replace(hour=23, minute=59, second=59, microsecond=999999),
                "description": "tomorrow",
            }
        if "this week" in q or "next week" in q:
            delta = 7 if "next week" in q else 0
            week_start = now + datetime.timedelta(days=delta)
            week_end   = week_start + datetime.timedelta(days=7)
            label = "next week" if "next week" in q else "this week"
            return {
                "start": week_start.replace(hour=0, minute=0, second=0, microsecond=0),
                "end":   week_end.replace(hour=23, minute=59, second=59, microsecond=999999),
                "description": label,
            }
        if "this month" in q or "next month" in q:
            if "next month" in q:
                if now.month == 12:
                    start = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    start = now.replace(month=now.month + 1, day=1)
            else:
                start = now.replace(day=1)
            end = start + datetime.timedelta(days=32)
            end = end.replace(day=1) - datetime.timedelta(seconds=1)
            label = "next month" if "next month" in q else "this month"
            return {"start": start, "end": end, "description": label}

        # Default: next 7 days
        return {
            "start": now,
            "end":   now + datetime.timedelta(days=7),
            "description": "in the next 7 days",
        }

    def _extract_fact_key(self, query: str) -> Optional[str]:
        """Map natural-language question to a user_profile key."""
        q = query.lower()
        KEY_MAP = {
            "favourite colour": "favourite_colour",
            "favorite colour":  "favourite_colour",
            "favourite color":  "favourite_colour",
            "favorite color":   "favourite_colour",
            "name":             "name",
            "address":          "address",
            "phone":            "phone_number",
            "email":            "email",
            "favourite food":   "favourite_food",
            "favorite food":    "favourite_food",
            "occupation":       "occupation",
            "job":              "occupation",
            "birthday":         "birthday",
            "age":              "age",
        }
        for keyword, key in KEY_MAP.items():
            if keyword in q:
                return key
        return None
