"""
KnowledgeExtractor — Background Worker for Layer 4 & Layer 5
=============================================================
One smart worker. One LLM call. Updates both layers simultaneously.

Triggered:
  - Every 20 messages in conversation (passive learning)
  - After Dreaming cycle completes (crystallization from dreams)
  - Via brain tool extract_knowledge_now() (immediate forced run)

Architecture:
  1. Read last 25 messages from session history
  2. Read existing L4 facts + L5 patterns (so LLM can update/delete, not just add)
  3. ONE structured LLM call → JSON response
  4. Parse: {facts: [...], patterns: [...]}
  5. Apply to SemanticMemory (L4) and ProceduralMemory (L5)
  6. Log what was learned silently — never interrupts user

Design principle (Tony Stark):
  One reactor. Not two. One LLM call returns everything we need.
  Zero impact on response latency — always runs as asyncio.create_task().
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM = """You are the deep learning subsystem of FRIDAY, an AI assistant.
Your job: analyze a conversation and extract structured knowledge about the user.

Extract TWO types of knowledge:

TYPE 1 — SEMANTIC FACTS: Stable, timeless truths about the user.
  Good examples: "studying Computer Science", "building an AI called FRIDAY", "lives in Hyderabad", "uses Windows 11"
  Bad examples: events with dates, one-time emotional states, temporary tasks, things the user asked FRIDAY to do

TYPE 2 — BEHAVIORAL PATTERNS: HOW the user communicates and works (not what they did, but HOW they prefer things).
  Good examples: "prefers bullet points for summaries", "wants code without explanation", "prefers concise answers"
  Bad examples: one-off requests, explicit commands, scheduled events

EXISTING PROFILE (Layer 6 — facts explicitly stated by the user. DO NOT re-add these to facts):
{existing_profile}

EXISTING FACTS (Layer 4 — check these before adding — use update/delete if needed):
{existing_facts}

EXISTING PATTERNS (Layer 5 — check these before adding — use update/delete if needed):
{existing_patterns}

INSTRUCTIONS:
- For each fact/pattern: decide if it should be added, updated, or deleted
- "add" → new knowledge NOT already in Profile, Facts, or Patterns above
- "update" → existing entry needs correction (use the 8-char id prefix shown in existing lists)
- "delete" → fact is no longer true or pattern was corrected
- If nothing meaningful was learned, return empty arrays
- Only extract things you are reasonably confident about (confidence >= 0.6)
- For subject in facts: always use the user's actual name if known, otherwise "the user"

OUTPUT — respond ONLY with valid JSON, nothing else:
{{
  "facts": [
    {{
      "action": "add",
      "subject": "Pranith",
      "predicate": "is studying",
      "object": "Computer Science, final year",
      "confidence": 0.8,
      "reason": "mentioned CS course multiple times"
    }},
    {{
      "action": "update",
      "id": "abc12345",
      "object": "updated value here",
      "confidence": 0.85,
      "reason": "user corrected previous fact"
    }},
    {{
      "action": "delete",
      "id": "xyz67890",
      "reason": "user said this is no longer true"
    }}
  ],
  "patterns": [
    {{
      "action": "add",
      "trigger": "asks for a summary",
      "behavior": "bullet points, maximum 5 items, no filler",
      "context": "communication",
      "confidence": 0.75,
      "reason": "user corrected to bullet format twice"
    }},
    {{
      "action": "update",
      "id": "def45678",
      "behavior": "corrected behavior description",
      "confidence": 0.8,
      "reason": "explicit correction by user"
    }},
    {{
      "action": "delete",
      "id": "ghi90123",
      "reason": "pattern no longer applies"
    }}
  ]
}}"""


class KnowledgeExtractor:
    """
    Background knowledge extractor for Layer 4 (Semantic) and Layer 5 (Procedural).

    Usage:
        extractor = KnowledgeExtractor(llm_provider, semantic_memory, procedural_memory)
        # In agent loop, every 20 messages:
        asyncio.create_task(extractor.run_once(recent_messages))
    """

    def __init__(self, llm_provider, semantic_memory, procedural_memory, profile=None):
        self.llm       = llm_provider
        self.l4        = semantic_memory
        self.l5        = procedural_memory
        self.l6        = profile    # Optional Layer 6 — used to prevent duplicate extraction
        self._running  = False  # prevents concurrent runs

    async def run_once(self, recent_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main extraction entry point.
        Fires ONE LLM call, updates both L4 and L5.
        Returns a summary of what was learned (for logging only).
        Never raises — all errors are caught and logged.
        """
        if self._running:
            logger.debug("[KnowledgeExtractor] Skipped — previous run still active")
            return {"facts_added": 0, "patterns_added": 0}

        self._running = True
        result = {"facts_added": 0, "facts_updated": 0, "facts_deleted": 0,
                  "patterns_added": 0, "patterns_updated": 0, "patterns_deleted": 0}
        try:
            # 1. Build conversation snippet (last 25 messages)
            conversation = self._format_messages(recent_messages)
            if not conversation.strip():
                return result

            # 2. Read existing L4 + L5 + L6 state so LLM can cross-check
            existing_facts    = self.l4.get_all_as_text() if self.l4 else "(none)"
            existing_patterns = self.l5.get_all_as_text() if self.l5 else "(none)"
            # L6 profile: prevents KnowledgeExtractor from re-adding what Reflection already stored
            existing_profile  = "(none)"
            if self.l6:
                try:
                    existing_profile = self.l6.get_context_string() or "(none)"
                except Exception:
                    pass

            # 3. Build prompt
            system_prompt = EXTRACTOR_SYSTEM.format(
                existing_profile  = existing_profile,
                existing_facts    = existing_facts,
                existing_patterns = existing_patterns,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Conversation to analyze:\n\n{conversation}"},
            ]

            # 4. LLM call
            raw = await self.llm.generate(messages, temperature=0.1)
            if not raw or not raw.strip():
                return result

            # 5. Parse
            data = self._safe_parse(raw)
            if not data:
                return result

            # 6. Apply to L4 (SemanticMemory)
            for f in data.get("facts", []):
                try:
                    action = f.get("action", "add")
                    if action == "add":
                        self.l4.add_fact(
                            subject    = f.get("subject", "the user"),
                            predicate  = f.get("predicate", ""),
                            object_value = f.get("object", ""),
                            confidence = float(f.get("confidence", 0.6)),
                            source     = "inferred",
                        )
                        result["facts_added"] += 1
                    elif action == "update" and f.get("id"):
                        # Find full id from the 8-char prefix
                        full_id = self._resolve_id_l4(f["id"])
                        if full_id:
                            self.l4.update_fact(
                                fact_id        = full_id,
                                new_object     = f.get("object"),
                                new_confidence = float(f["confidence"]) if "confidence" in f else None,
                            )
                            result["facts_updated"] += 1
                    elif action == "delete" and f.get("id"):
                        full_id = self._resolve_id_l4(f["id"])
                        if full_id:
                            self.l4.delete_fact(full_id)
                            result["facts_deleted"] += 1
                except Exception as fe:
                    logger.debug(f"[KnowledgeExtractor] Fact op error: {fe}")

            # 7. Apply to L5 (ProceduralMemory)
            for p in data.get("patterns", []):
                try:
                    action = p.get("action", "add")
                    if action == "add":
                        await self.l5.add_pattern(
                            trigger    = p.get("trigger", ""),
                            behavior   = p.get("behavior", ""),
                            context    = p.get("context", "general"),
                            confidence = float(p.get("confidence", 0.5)),
                        )
                        result["patterns_added"] += 1
                    elif action == "update" and p.get("id"):
                        pat = await self.l5.get_pattern(
                            trigger = self._resolve_trigger_l5(p["id"]),
                            context = p.get("context", "general"),
                        )
                        if pat:
                            await self.l5.correct_pattern(
                                trigger      = pat.key,
                                old_behavior = pat.value,
                                new_behavior = p.get("behavior", pat.value),
                                context      = pat.context,
                            )
                            result["patterns_updated"] += 1
                    elif action == "delete" and p.get("id"):
                        trigger = self._resolve_trigger_l5(p["id"])
                        if trigger:
                            await self.l5.delete_pattern(trigger)
                            result["patterns_deleted"] += 1
                except Exception as pe:
                    logger.debug(f"[KnowledgeExtractor] Pattern op error: {pe}")

            logger.info(
                f"[KnowledgeExtractor] Done — "
                f"facts: +{result['facts_added']} ~{result['facts_updated']} -{result['facts_deleted']} | "
                f"patterns: +{result['patterns_added']} ~{result['patterns_updated']} -{result['patterns_deleted']}"
            )

        except Exception as e:
            logger.warning(f"[KnowledgeExtractor] Extraction failed (non-fatal): {e}")
        finally:
            self._running = False

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format raw message history into readable conversation text."""
        lines = []
        for msg in messages[-25:]:  # cap at 25
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                prefix = "User" if role == "user" else "FRIDAY"
                lines.append(f"{prefix}: {content.strip()[:400]}")
        return "\n".join(lines)

    def _safe_parse(self, raw: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response, handling markdown fences."""
        text = raw.strip()
        # Strip markdown code fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        # Find JSON object boundaries
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.debug(f"[KnowledgeExtractor] JSON parse error: {e}")
            return None

    def _resolve_id_l4(self, short_id: str) -> Optional[str]:
        """Find a full fact UUID given the 8-char prefix shown in existing_facts text."""
        if not self.l4:
            return None
        try:
            facts = self.l4.get_all_active(limit=100)
            for f in facts:
                if f["id"].startswith(short_id):
                    return f["id"]
        except Exception:
            pass
        return None

    def _resolve_trigger_l5(self, short_id: str) -> Optional[str]:
        """Find the trigger string given the 8-char pattern ID prefix."""
        if not self.l5:
            return None
        try:
            all_p = self.l5.engine.get_all_patterns(limit=100)
            for p in all_p:
                if p.pattern_id.startswith(short_id):
                    return p.key
        except Exception:
            pass
        return None
