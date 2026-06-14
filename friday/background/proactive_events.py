import os
import json
import asyncio
from datetime import datetime
from friday.llm.base import LLMProvider
import logging

logger = logging.getLogger(__name__)

class ProactiveEventsWatcher:
    def __init__(self, workspace_dir: str):
        self.state_path = os.path.join(workspace_dir, "alerts_state.json")
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                    # Migrate old state files that don't have alerted_now
                    if "alerted_now" not in data:
                        data["alerted_now"] = []
                    return data
            except Exception:
                pass
        return {"alerted_1h": [], "alerted_15m": [], "alerted_now": []}

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.warning(f"Could not save alert state: {e}")

    async def check(self, db_manager, config, llm_provider: LLMProvider = None) -> str:
        """
        Checks 3 alert windows per event:
          - 1h  window : 45-65 mins before start  (floor lowered so same-hour adds still fire)
          - 15m window : 10-20 mins before start
          - NOW window : -2 to +2 mins (fires at the exact moment the event begins)

        Each event fires each window at most once (deduped via alerts_state.json).
        Bypasses AgentLoop entirely - direct LLM call for minimal latency.
        Future-proof: swap config.llama_model for Gemini/TTS output with no other changes.
        """
        conn = db_manager.get_connection()
        now = datetime.now()

        rows = conn.execute("SELECT * FROM facts WHERE status = 'active'").fetchall()

        alerts_needed = []

        for row in rows:
            fact_id = row["id"]
            content = row["content"]
            try:
                iso_str = row["date_start"].replace("Z", "+00:00")
                start_dt = datetime.fromisoformat(iso_str)
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.astimezone().replace(tzinfo=None)
            except Exception:
                continue

            mins_until = (start_dt - now).total_seconds() / 60.0

            # Window 1: 1 hour (floor lowered to 45 so same-hour events still fire)
            if 45 <= mins_until <= 65 and fact_id not in self.state["alerted_1h"]:
                alerts_needed.append({
                    "id": fact_id,
                    "content": content,
                    "time": f"{int(mins_until)} minutes",
                    "urgency": "1h"
                })
                self.state["alerted_1h"].append(fact_id)

            # Window 2: 15 minutes
            elif 10 <= mins_until <= 20 and fact_id not in self.state["alerted_15m"]:
                alerts_needed.append({
                    "id": fact_id,
                    "content": content,
                    "time": f"{int(mins_until)} minutes",
                    "urgency": "15m"
                })
                self.state["alerted_15m"].append(fact_id)

            # Window 3: RIGHT NOW (exact time, -2 to +2 min buffer for scheduler jitter)
            elif -2 <= mins_until <= 2 and fact_id not in self.state["alerted_now"]:
                alerts_needed.append({
                    "id": fact_id,
                    "content": content,
                    "time": "NOW",
                    "urgency": "now"
                })
                self.state["alerted_now"].append(fact_id)

        if not alerts_needed:
            return None

        self._save_state()

        # Build urgency-aware prompt - different tone per window
        now_events    = [a for a in alerts_needed if a["urgency"] == "now"]
        urgent_events = [a for a in alerts_needed if a["urgency"] == "15m"]
        soon_events   = [a for a in alerts_needed if a["urgency"] == "1h"]

        events_str = "\n".join(
            [f"- [HAPPENING NOW] {a['content']}" for a in now_events] +
            [f"- [IN {a['time']}] {a['content']}" for a in urgent_events] +
            [f"- [IN ~1 HOUR] {a['content']}" for a in soon_events]
        )

        prompt = f"""You are Friday, a JARVIS-like AI assistant running as a background worker.

Upcoming events requiring attention:
{events_str}

Rules:
- Write exactly ONE sentence per [HAPPENING NOW] event - highest priority.
- If multiple events, combine into max 2 sentences total.
- Address formally as Sir.
- No markdown, no lists. Pure urgent prose only.
- For [HAPPENING NOW]: use language like "is starting now" or "has begun".
- For [IN X MINUTES]: say "in X minutes".
- For [IN ~1 HOUR]: say "in about an hour"."""

        try:
            if llm_provider:
                messages = [{"role": "user", "content": prompt}]
                return await llm_provider.generate(messages, temperature=0.3)
            # Legacy direct ollama fallback
            import ollama as _ollama
            client = _ollama.AsyncClient()
            response = await client.chat(
                model=config.llama_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            # Hardcoded fallback - never silently miss an alert even if LLM is down
            top = alerts_needed[0]
            if top["urgency"] == "now":
                return f"Sir, {top['content']} is starting right now."
            return f"Sir, {top['content']} starts in {top['time']}."
