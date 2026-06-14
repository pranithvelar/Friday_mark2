import json
import re
import os
import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional
from friday.llm.base import LLMProvider
import datetime
from friday.memory.layers.layer_3_episodic import FactStore, extract_facts
from friday.agents.friday.session_repair import repair_tool_use_result_pairing, extract_identifiers

LLM_TIMEOUT_SECONDS = 60
NUM_CTX = 8192  # Sweet spot: 4x Ollama default, negligible latency impact

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Token-aware context assembly (inspired by OpenClaw's context engine)
# -----------------------------------------------------------------------
# With NUM_CTX=8192, we budget:
#   System prompt + tools: ~800 tokens (minimized)
#   Summary (if any):      ~300 tokens
#   Recent messages:       ~5500 tokens (the MAIN content)
#   Response headroom:     ~1600 tokens
# -----------------------------------------------------------------------

MAX_CONTEXT_TOKENS = 6000
RESPONSE_RESERVE = 1600
SUMMARY_MAX_TOKENS = 300
MIN_SUMMARIZE_THRESHOLD = 20
CHARS_PER_TOKEN = 4
OVERSIZED_MSG_THRESHOLD = 800  # tokens — messages above this are excluded in Stage 2
COMPACTION_RETRY_ATTEMPTS = 3  # retry per chunk in Stage 1


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Dict[str, str]) -> int:
    return estimate_tokens(msg.get("content", "")) + 4


class FeedbackDetector:
    PATTERNS = [
        (r'(?:call me|address me as|my name is|i am)\s+([\w]+)', 'address_as', 1),
        (r'(?:be more|be)\s+(concise|brief|short|verbose|detailed|formal|casual|friendly)', 'response_style', 1),
        (r'(?:keep.+(?:short|brief|concise))', 'response_style', 'concise'),
        (r'(?:too long|too verbose|shorten)', 'response_style', 'concise'),
        (r'(?:be more|sound more)\s+(serious|funny|playful|warm|professional|chill)', 'tone', 1),
        (r"(?:don'?t use)\s+(?:emojis?)", 'use_emojis', 'no'),
        (r'(?:use)\s+(?:emojis?)', 'use_emojis', 'yes'),
    ]

    def __init__(self, personalization):
        self.personalization = personalization
        self._compiled = [(re.compile(p, re.IGNORECASE), k, v) for p, k, v in self.PATTERNS]

    def detect_and_save(self, user_message: str) -> list:
        if not self.personalization:
            return []
        saved = []
        for pattern, pref_key, value_spec in self._compiled:
            match = pattern.search(user_message)
            if match:
                value = match.group(value_spec) if isinstance(value_spec, int) else value_spec
                value = value.strip()
                current = self.personalization.get_preference(pref_key)
                if current != value:
                    self.personalization.update_preference(pref_key, value)
                    saved.append((pref_key, value))
        return saved


SYSTEM_PROMPT = """You are Friday, a highly intelligent, formal, and concise personal AI assistant with persistent memory.

CRITICAL INTELLIGENCE RULES:
1. [ABSOLUTE CONTINUOUS ITINERARY] is YOUR internal awareness - NEVER list it unless explicitly asked "what's my schedule".
2. Use [Memory context] to answer smartly but NEVER repeat it back.
3. Answer ONLY the current question. Keep responses 1-2 sentences.
4. NEVER mention tags, tools, warnings, or that you are an AI.
5. EXECUTOR: When user asks to DO something, output tool JSON immediately.
6. LEARNING: Silently save stable facts/preferences with tools. No confirmation messages.
7. If [SYSTEM WARNING] about conflicts exists, mention it ONCE briefly.
8. TEMPORAL ACCURACY: For schedules and upcoming events, rely EXCLUSIVELY on [ABSOLUTE CONTINUOUS ITINERARY]. Ignore [Memory context] for dates, as it contains historical/outdated snippets.
{user_rules}

To use a tool, output ONLY a JSON block:
```json
{{"name": "tool_name", "arguments": {{"arg": "val"}}}}
```
Then STOP.

Tools:
{tools_schema}
"""


class AgentLoop:
    def __init__(self, workspace_dir: str = "", model: str = "llama3.1:8b", session_manager=None,
                 session_id: str = "default", personalization=None, db_manager=None,
                 llm_provider: Optional[LLMProvider] = None):
        self.workspace_dir = workspace_dir
        self.db_manager = db_manager
        self.fact_store = FactStore(db_manager) if db_manager else None
        self.model = model  # kept for backward compat / logging
        self._llm_provider = llm_provider  # injected at startup
        self.tools: Dict[str, Callable] = {}
        self.tools_schemas: List[Dict[str, Any]] = []
        self.session_manager = session_manager
        self.session_id = session_id
        self.personalization = personalization
        # FeedbackDetector is kept as a lightweight fallback for instant patterns
        # but LLM auto-tooling now handles the majority of preference learning.
        self.feedback_detector = FeedbackDetector(personalization) if personalization else None
        self._status_callback = None
        self._history: List[Dict[str, str]] = []
        self._history_loaded = False
        self._summary_cache: str = ""
        self._compacted_up_to: int = 0
        self._reflect_at: int = 0  # tracks when next Reflection should run
        self._searcher = None  # injected by terminal_chat for pre-search

    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any]):
        self.tools[name] = func
        self.tools_schemas.append(schema)

    def _build_system_prompt(self) -> str:
        tools_brief = []
        for s in self.tools_schemas:
            params = s.get("parameters", {}).get("properties", {})
            param_list = ", ".join(f'{k}: {v.get("type","str")}' for k, v in params.items())
            tools_brief.append(f'- {s["name"]}({param_list}): {s.get("description","")[:80]}')
        tools_str = "\n".join(tools_brief)

        user_rules = ""
        if self.personalization:
            prefs = self.personalization.profile.get("preferences", {})
            rules = []
            if prefs.get("address_as"):
                rules.append(f'- Address the user as "{prefs["address_as"]}".')
            if prefs.get("response_style") == "concise":
                rules.append("- Be very concise.")
            if prefs.get("tone"):
                rules.append(f"- Use a {prefs['tone']} tone.")
            if prefs.get("use_emojis") == "no":
                rules.append("- No emojis.")
            if rules:
                user_rules = "\n".join(rules) + "\n"
                
        # Fetch precise timezone-aware time to prevent Windows OS clock drift
        local_now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        now_str = local_now.strftime("%A, %B %d, %Y, %I:%M %p %Z").strip()
        time_injection = f"\nCURRENT SYSTEM TIME: {now_str}\n"

        return SYSTEM_PROMPT.format(tools_schema=tools_str, user_rules=user_rules) + time_injection

    def _load_history(self):
        if self._history_loaded or not self.session_manager:
            return
        prior = self.session_manager.load_session(self.session_id)
        if prior:
            self._history = prior
        self._load_summary_cache()
        self._history_loaded = True

    def _persist(self, role: str, content: str):
        msg = {"role": role, "content": content}
        self._history.append(msg)
        if self.session_manager:
            self.session_manager.append_message(self.session_id, role, content)

    def _get_summary_key(self) -> str:
        return f"summary:{self.session_id}"

    def _load_summary_cache(self):
        if not self.db_manager:
            return
        try:
            conn = self.db_manager.get_connection()
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (self._get_summary_key(),)
            ).fetchone()
            if row:
                self._summary_cache = row["value"][:SUMMARY_MAX_TOKENS * CHARS_PER_TOKEN]
                if self._summary_cache:
                    self._compacted_up_to = max(0, len(self._history) - 10)
        except Exception:
            pass

    def _save_summary_cache(self, summary: str):
        self._summary_cache = summary[:SUMMARY_MAX_TOKENS * CHARS_PER_TOKEN]
        if not self.db_manager:
            return
        try:
            conn = self.db_manager.get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (self._get_summary_key(), self._summary_cache)
            )
            conn.commit()
        except Exception:
            pass

    def _status(self, msg: str):
        if self._status_callback:
            self._status_callback(msg)

    async def _llm_call(self, messages: list) -> str:
        """Route LLM call through the slotted provider (API → Ollama fallback)."""
        if self._llm_provider:
            return await self._llm_provider.generate(
                messages,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        # Safety fallback: if no provider injected, use ollama directly (legacy mode)
        try:
            import ollama
            client = ollama.AsyncClient()
            response = await asyncio.wait_for(
                client.chat(
                    model=self.model,
                    messages=messages,
                    options={"num_ctx": NUM_CTX}
                ),
                timeout=LLM_TIMEOUT_SECONDS
            )
            return response['message']['content']
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"LLM timed out after {LLM_TIMEOUT_SECONDS}s")

    def _assemble_context(self, user_message: str) -> List[Dict[str, str]]:
        """Token-aware context assembly. Current question always fits."""
        system_prompt = self._build_system_prompt()
        user_msg_tokens = estimate_tokens(user_message) + 4
        budget = MAX_CONTEXT_TOKENS - user_msg_tokens

        messages = [{"role": "system", "content": system_prompt}]

        summary_tokens = 0
        if self._summary_cache:
            summary_tokens = min(estimate_tokens(self._summary_cache), SUMMARY_MAX_TOKENS)

        history_budget = budget - summary_tokens
        recent_msgs = []
        tokens_used = 0

        for msg in reversed(self._history):
            msg_tokens = estimate_message_tokens(msg)
            if tokens_used + msg_tokens > history_budget:
                break
            recent_msgs.append(msg)
            tokens_used += msg_tokens

        recent_msgs.reverse()

        if self._summary_cache and summary_tokens > 0:
            messages.append({
                "role": "user",
                "content": f"[Previous context: {self._summary_cache}]"
            })
            messages.append({
                "role": "assistant",
                "content": "Got it."
            })

        messages.extend(recent_msgs)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _maybe_compact(self):
        """Compact only when enough new messages accumulate.
        
        Uses multi-stage progressive fallback (inspired by OpenClaw):
          Stage 1: Full chunked summarization with identifier preservation
          Stage 2: Partial (exclude oversized messages, note them)
          Stage 3: Hard text fallback — guaranteed never to crash
        
        After compaction, repairs tool-use pairing on retained messages
        to prevent orphaned tool_results from causing API errors.
        """
        total = len(self._history)
        uncompacted = total - self._compacted_up_to
        if uncompacted < MIN_SUMMARIZE_THRESHOLD:
            return

        self._status("Summarizing older context...")
        keep_recent = 8
        old_end = max(self._compacted_up_to, total - keep_recent)
        old_messages = self._history[self._compacted_up_to:old_end]
        if not old_messages:
            return

        # --- Repair tool-use pairing on the RETAINED recent messages ---
        flat_rest = self._history[old_end:]
        repair_report = repair_tool_use_result_pairing(flat_rest)
        if repair_report["repaired"]:
            logger.info(f"Compaction boundary repair: {repair_report['stats']}")
            self._history = self._history[:old_end] + repair_report["messages"]

        # --- Multi-stage summarization ---
        summary = await self._summarize_in_stages(old_messages)
        if summary:
            self._save_summary_cache(summary.strip())
            self._compacted_up_to = old_end

    async def _summarize_in_stages(self, messages: List[Dict[str, str]]) -> str:
        """Multi-stage progressive compaction with fallback.
        
        Stage 1: Full summarization with token-balanced chunks and retries.
                  Preserves opaque identifiers (UUIDs, hashes, IPs, URLs, file paths).
        Stage 2: Exclude oversized messages, summarize the rest.
        Stage 3: Hard text fallback — never crashes.
        """
        # --- Stage 1: Full chunked summarization ---
        try:
            summary = await self._stage1_full_summarize(messages)
            if summary:
                logger.info("Compaction Stage 1 succeeded (full summarization)")
                return summary
        except Exception as e:
            logger.warning(f"Compaction Stage 1 failed: {e}")

        # --- Stage 2: Partial (exclude oversized) ---
        try:
            summary = await self._stage2_partial_summarize(messages)
            if summary:
                logger.info("Compaction Stage 2 succeeded (partial summarization)")
                return summary
        except Exception as e:
            logger.warning(f"Compaction Stage 2 failed: {e}")

        # --- Stage 3: Hard text fallback (never crashes) ---
        logger.warning("Compaction falling back to Stage 3 (hard text fallback)")
        return self._stage3_hard_fallback(messages)

    async def _stage1_full_summarize(self, messages: List[Dict[str, str]]) -> str:
        """Stage 1: Split into token-balanced chunks, summarize each, merge."""
        # Collect all identifiers for preservation verification
        all_identifiers = set()
        for msg in messages:
            all_identifiers.update(extract_identifiers(msg.get("content", "")))

        # Build chunks of ~1500 tokens each
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_limit = 1500

        for msg in messages:
            msg_tokens = estimate_message_tokens(msg)
            if current_tokens + msg_tokens > chunk_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(msg)
            current_tokens += msg_tokens

        if current_chunk:
            chunks.append(current_chunk)

        # Summarize each chunk with retries
        chunk_summaries = []
        for chunk in chunks:
            content_lines = []
            if self._summary_cache and not chunk_summaries:
                content_lines.append(f"Previous context: {self._summary_cache}")
            for msg in chunk:
                role = msg.get("role", "?")
                text = msg.get("content", "")[:300]
                content_lines.append(f"[{role}]: {text}")

            summary = None
            for attempt in range(COMPACTION_RETRY_ATTEMPTS):
                try:
                    summary = await self._llm_call([
                        {"role": "system", "content": (
                            "Summarize in 3-4 bullet points. Keep facts, names, preferences. "
                            "Be extremely concise. Output ONLY bullets.\n\n"
                            "CRITICAL: Preserve all opaque identifiers exactly as written "
                            "(no shortening or reconstruction), including UUIDs, hashes, IDs, "
                            "hostnames, IPs, ports, URLs, and file names."
                        )},
                        {"role": "user", "content": "\n".join(content_lines)}
                    ])
                    if summary and summary.strip():
                        break
                except Exception as e:
                    logger.warning(f"Chunk summarization attempt {attempt+1} failed: {e}")
                    if attempt < COMPACTION_RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(0.5)

            if summary and summary.strip():
                chunk_summaries.append(summary.strip())

        if not chunk_summaries:
            return ""

        # Merge chunk summaries if multiple
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        merge_content = "\n\n".join(
            f"Chunk {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        merged = await self._llm_call([
            {"role": "system", "content": (
                "Merge these chunk summaries into a single concise summary (3-5 bullets). "
                "Keep all facts, names, and identifiers. Output ONLY bullets.\n\n"
                "CRITICAL: Preserve all opaque identifiers exactly as written "
                "(no shortening or reconstruction), including UUIDs, hashes, IDs, "
                "hostnames, IPs, ports, URLs, and file names."
            )},
            {"role": "user", "content": merge_content}
        ])
        return merged.strip() if merged else ""

    async def _stage2_partial_summarize(self, messages: List[Dict[str, str]]) -> str:
        """Stage 2: Exclude oversized messages, note them, summarize the rest."""
        normal_msgs = []
        oversized_notes = []

        for msg in messages:
            msg_tokens = estimate_message_tokens(msg)
            if msg_tokens > OVERSIZED_MSG_THRESHOLD:
                role = msg.get("role", "?")
                preview = msg.get("content", "")[:80]
                oversized_notes.append(f"[{role} message, ~{msg_tokens} tokens]: {preview}...")
            else:
                normal_msgs.append(msg)

        if not normal_msgs:
            # All messages are oversized — fall through to Stage 3
            return ""

        content_lines = []
        if self._summary_cache:
            content_lines.append(f"Previous context: {self._summary_cache}")
        for msg in normal_msgs[-15:]:
            role = msg.get("role", "?")
            text = msg.get("content", "")[:150]
            content_lines.append(f"[{role}]: {text}")

        if oversized_notes:
            content_lines.append(f"\n[Note: {len(oversized_notes)} oversized messages excluded]")

        summary = await self._llm_call([
            {"role": "system", "content": (
                "Summarize in 3-4 bullet points. Keep facts, names, preferences. "
                "Be extremely concise. Output ONLY bullets.\n\n"
                "CRITICAL: Preserve all opaque identifiers exactly as written."
            )},
            {"role": "user", "content": "\n".join(content_lines)}
        ])
        result = summary.strip() if summary else ""
        if oversized_notes:
            result += f"\n[{len(oversized_notes)} oversized messages were excluded from summary]"
        return result

    def _stage3_hard_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Stage 3: Hard text fallback — guaranteed never to crash."""
        oversized_notes = []
        for msg in messages:
            if estimate_message_tokens(msg) > OVERSIZED_MSG_THRESHOLD:
                oversized_notes.append(msg.get("content", "")[:40])

        return (
            f"Context contained {len(messages)} messages"
            f" ({len(oversized_notes)} oversized)."
            f" Summary unavailable due to size limits."
        )

    def _fix_json_keys(self, text: str) -> str:
        fixed = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1 "\2":', text)
        for tn in self.tools.keys():
            fixed = re.sub(rf'"name"\s*:\s*{re.escape(tn)}', f'"name": "{tn}"', fixed)
        return fixed

    def _extract_action(self, text: str) -> Optional[Dict[str, Any]]:
        if "```json" in text:
            try:
                json_str = text.split("```json")[-1].split("```")[0].strip()
                json_str = self._fix_json_keys(json_str)
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "name" in parsed:
                    return parsed
            except Exception:
                pass

        if "{" in text and "name" in text:
            depth = 0
            start = -1
            for i, c in enumerate(text):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidate = text[start:i+1]
                        try:
                            fixed = self._fix_json_keys(candidate)
                            parsed = json.loads(fixed)
                            if isinstance(parsed, dict) and "name" in parsed:
                                return parsed
                        except Exception:
                            pass
                        start = -1
        return None

    async def _pre_search(self, user_message: str) -> str:
        """Auto-search memory before every response — proactive conflict detection."""
        if not self._searcher:
            return ""
        try:
            results = await self._searcher.search(
                user_message, vector_weight=0.5, text_weight=0.5, max_results=4
            )
            if not results:
                return ""
            lines = []
            for r in results:
                snippet = r.snippet[:200] if r.snippet else ""
                lines.append(f"- {snippet}")
            return "[Memory context]\n" + "\n".join(lines)
        except Exception as e:
            logger.warning(f"Pre-search failed: {e}")
            return ""

    async def _maybe_reflect(self):
        """Background Reflection agent: every 12 messages, silently scans chat history
        for new stable facts/preferences and autonomously saves them."""
        total = len(self._history)
        if total < self._reflect_at + 12:
            return
        if not self.personalization:
            return

        self._reflect_at = total
        recent = self._history[-14:]
        if not recent:
            return

        content_lines = []
        for msg in recent:
            role = msg.get("role", "?")
            text = msg.get("content", "")[:200]
            content_lines.append(f"[{role}]: {text}")

        reflection_prompt = (
            "Analyze the following conversation. Extract ONLY deliberate, stable personal facts "
            "or preferences expressed by the user. Ignore one-time emotional states. "
            "Output ONLY a compact JSON array like: "
            '[{{"type":"fact","key":"occupation","value":"designer"}}, '
            '{{"type":"preference","key":"tone","value":"casual"}}]. '
            "If nothing stable was revealed, output an empty array []."
            "\n\nConversation:\n" + "\n".join(content_lines)
        )

        try:
            import json as _json
            result_text = await self._llm_call([
                {"role": "system", "content": "You are a silent observer extracting stable user attributes from conversation logs. Output ONLY valid JSON."},
                {"role": "user", "content": reflection_prompt}
            ])
            # Extract JSON from response
            text = result_text.strip()
            if "[" in text:
                text = text[text.index("["):text.rindex("]")+1]
            items = _json.loads(text)
            for item in items:
                t = item.get("type", "")
                key = item.get("key", "").strip()
                value = item.get("value", "").strip()
                if not key or not value:
                    continue
                if t == "fact":
                    self.personalization.update_fact(key, value)
                    logger.info(f"[Reflection] Saved fact: {key}={value}")
                elif t == "preference":
                    self.personalization.update_preference(key, value)
                    logger.info(f"[Reflection] Saved preference: {key}={value}")
        except Exception as e:
            logger.debug(f"Reflection agent failed silently: {e}")

    async def run(self, user_message: str, max_steps: int = 5) -> str:
        # Acquire session write-lock to prevent concurrent corruption
        if self.session_manager:
            lock = self.session_manager.get_lock(self.session_id)
        else:
            lock = asyncio.Lock()  # dummy lock if no session manager

        async with lock:
            return await self._run_locked(user_message, max_steps)

    async def _run_locked(self, user_message: str, max_steps: int = 5) -> str:
        """Main agent loop body, runs under session write-lock."""
        self._load_history()

        # FeedbackDetector: still handles instant, obvious patterns (name, emojis)
        # The LLM auto-tooling (LEARNING PROTOCOL rule) handles everything natural
        if self.feedback_detector:
            self.feedback_detector.detect_and_save(user_message)

        # Compaction runs AFTER the response is returned (non-blocking).
        # This eliminates the "[Summarizing older context...]" delay on the hot path.
        # The summary will be ready in the DB cache before the NEXT user message.

        # Background Reflection: silently learns from conversation every 12 messages
        await self._maybe_reflect()

        # Auto-search memory BEFORE LLM call — injects relevant memories as context
        memory_context = await self._pre_search(user_message)
        
        # EXTRACT FACTS AND CHECK CONFLICTS
        active_facts_context = ""
        system_conflict_warning = ""
        reminder_context = ""
        if self.fact_store:
            # 1. Extract ops blind to context
            extraction_res = extract_facts(user_message)
            ops = extraction_res.get("operations", [])

            # Apply ops safely behind the scenes
            for op in ops:
                try:
                    if op.get("action") == "add" and op.get("content") and op.get("date_start"):
                        start_dt = datetime.datetime.fromisoformat(op["date_start"])
                        end_dt = datetime.datetime.fromisoformat(op["date_end"]) if op.get("date_end") else start_dt
                        self.fact_store.add_fact(
                            op["content"], 
                            start_dt, 
                            end_dt,
                            float(op.get("importance", 0.5))
                        )
                    elif op.get("action") == "delete" and op.get("keyword"):
                        self.fact_store.delete_fact(op["keyword"])
                except Exception as e:
                    logger.warning(f"Error applying fact operation {op}: {e}")

            # 2. Run the offline mathematical linter
            self.fact_store.lint_memory_conflicts()

            # 3. Warn about ALL contested future events — no urgency gate.
            #    Every conflict matters, whether it's tomorrow or 4 months out.
            contested = self.fact_store.get_contested_facts()
            if contested:
                now_dt = datetime.datetime.now()
                lines = []
                for f in contested:
                    try:
                        ds = datetime.datetime.fromisoformat(f["date_start"]).replace(tzinfo=None)
                        days_until = (ds.date() - now_dt.date()).days
                        if days_until < 0:
                            continue  # already expired, skip
                        if days_until == 0:
                            label = "TODAY"
                        elif days_until == 1:
                            label = "TOMORROW"
                        else:
                            label = f"{days_until} days away"
                        f_date = ds.strftime("%a %b %d")
                        lines.append(f"- {f_date} ({label}): {f['content']}")
                    except Exception:
                        pass
                if lines:
                    system_conflict_warning = (
                        "[SYSTEM WARNING: CONFLICTING EVENTS DETECTED — "
                        "these events have overlapping times and must be resolved:]\n"
                        + "\n".join(lines) + "\n"
                    )

            # 4. Build the full infinite-horizon itinerary with countdown labels.
            #    The LLM sees ALL future events and how many days remain —
            #    rule #1 in the system prompt keeps it silent unless asked.
            current_active = self.fact_store.get_active_facts()  # no day cap
            if current_active:
                now_dt = datetime.datetime.now()
                lines = []
                for f in current_active:
                    try:
                        ds = datetime.datetime.fromisoformat(f["date_start"]).replace(tzinfo=None)
                        de = datetime.datetime.fromisoformat(f["date_end"]).replace(tzinfo=None)
                        days_until = (ds.date() - now_dt.date()).days

                        if days_until < 0:
                            day_label = f"ONGOING (started {-days_until} days ago)"
                        elif days_until == 0:
                            day_label = f"TODAY {ds.strftime('%I:%M %p')}-{de.strftime('%I:%M %p')}"
                        elif days_until == 1:
                            day_label = f"TOMORROW {ds.strftime('%I:%M %p')}"
                        else:
                            day_label = f"{ds.strftime('%a %b %d')} ({days_until} days away)"

                        lines.append(f"- {day_label}: {f['content']}")
                    except Exception:
                        pass
                if lines:
                    active_facts_context = (
                        "[ABSOLUTE CONTINUOUS ITINERARY — ALL UPCOMING EVENTS]\n"
                        + "\n".join(lines) + "\n"
                    )

            # 5. Day-before reminder — inject a prominent block for events
            #    starting in the next 20-48 hours that haven't been reminded yet.
            #    Unlike the silent itinerary, the LLM IS expected to mention this.
            reminder_context = ""
            if self.fact_store:
                try:
                    reminder_events = self.fact_store.get_events_needing_reminder()
                    if reminder_events:
                        reminder_lines = []
                        for rev in reminder_events:
                            try:
                                ds = datetime.datetime.fromisoformat(rev["date_start"]).replace(tzinfo=None)
                                de = datetime.datetime.fromisoformat(rev["date_end"]).replace(tzinfo=None)
                                now_dt = datetime.datetime.now()
                                hours_until = max(0, int((ds - now_dt).total_seconds() // 3600))
                                reminder_lines.append(
                                    f"- {ds.strftime('%A %b %d at %I:%M %p')} "
                                    f"(in ~{hours_until} hours): {rev['content']}"
                                )
                                # Mark as reminded so this never fires again for this event
                                self.fact_store.mark_reminder_sent(rev["id"])
                            except Exception:
                                pass
                        if reminder_lines:
                            reminder_context = (
                                "[\U0001f514 REMINDER — UPCOMING TOMORROW]\n"
                                "These events start within the next 24-48 hours. "
                                "Proactively mention this to the user.\n"
                                + "\n".join(reminder_lines) + "\n"
                            )
                except Exception as e:
                    logger.warning(f"Reminder check failed: {e}")

        # Prepend memory context to the user message so the LLM always sees it
        augmented_message = user_message
        context_parts = []
        if reminder_context:                  # 🔔 Reminders first — highest priority
            context_parts.append(reminder_context)
        if active_facts_context:
            context_parts.append(active_facts_context)
        if system_conflict_warning:
            context_parts.append(system_conflict_warning)
        if memory_context:
            context_parts.append(memory_context)
            
        if context_parts:
            augmented_message = "\n".join(context_parts) + f"\nUser: {user_message}"

        messages = self._assemble_context(augmented_message)

        self._persist("user", user_message)  # Persist original (not augmented)
        self._status("Generating response...")

        response_content = None
        for step in range(max_steps):
            try:
                content = await self._llm_call(messages)
                messages.append({"role": "assistant", "content": content})

                action = self._extract_action(content)
                if not action:
                    self._persist("assistant", content)
                    response_content = content
                    break

                self._persist("assistant", content)
                tool_name = action.get("name")
                tool_args = action.get("arguments", {})

                if tool_name in self.tools:
                    try:
                        result = await self.tools[tool_name](**tool_args)
                        result_msg = f"Result: {result}"
                    except Exception as e:
                        result_msg = f"Result: Tool {tool_name} failed: {e}"
                else:
                    result_msg = f"Result: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"

                self._persist("user", result_msg)
                messages.append({"role": "user", "content": result_msg})

            except asyncio.TimeoutError:
                response_content = "Response timed out. Please try again."
                break
            except Exception as e:
                logger.error(f"Step {step} error: {type(e).__name__}: {e}")
                response_content = f"Error: {type(e).__name__}. Please try again."
                break

        if response_content is None:
            response_content = "Max reasoning steps reached."

        # Fire compaction in the background AFTER returning the response.
        # Zero impact on response latency. Summary cached for next turn.
        try:
            from friday.background.context_summarizer import trigger_background_compact
            trigger_background_compact(self)
        except Exception:
            pass  # Never let import/task errors affect the response

        return response_content
