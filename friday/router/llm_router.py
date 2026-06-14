"""
LLM Router — Intelligent Tier Classifier
==========================================
Replaces the hardcoded regex classifier with a real LLM call that
understands context, intent, and nuance.

Design:
  - Makes ONE ultra-fast constrained LLM call (output = 1 word)
  - Hard 2-second timeout → falls back to regex silently
  - Result is cached per session so the same query never hits LLM twice
  - Regex FastIntentClassifier is ALWAYS the safety net — never fails

Routing decision:
  "simple"  → Direct DB/memory lookup. No tool needed. Sub-200ms response.
  "medium"  → 1-3 tool calls. AgentLoop runs with max_steps=3.
  "complex" → Multi-step plan needed. MultiAgentPlanner + ExecutionEngine.

Why LLM routing is better than regex:
  - Regex: "what conflicting event?" → short query → SIMPLE (wrong)
  - LLM:   "what conflicting event?" → needs context lookup → MEDIUM (correct)
  - Regex: "remind me of something" → matches SINGLE_ACTION → MEDIUM
  - LLM:   "remind me of something" → vague, conversational → SIMPLE (correct)
  - Regex: Can't understand user's tone, context, or ambiguity
  - LLM:   Understands that "ugh just do it" is different from "research X"
"""

import asyncio
import logging
import re
from typing import Tuple, Optional, Dict
import ollama

from friday.router.intent_classifier import FastIntentClassifier, QueryComplexity, QueryCategory
from friday.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# ── Timing constants ──────────────────────────────────────────────────────────
ROUTER_LLM_TIMEOUT = 2.0   # Max seconds to wait for routing decision (fast fail → regex)
ROUTER_NUM_CTX     = 256   # Tiny context — just the query + system prompt

# ── The classification prompt ─────────────────────────────────────────────────
# This is intentionally minimal. We want ONE word out, nothing else.
_ROUTER_SYSTEM = """\
You are a routing classifier for an AI assistant called Friday.
Classify the user's message into exactly ONE of these three tiers:

simple  - Greetings, status checks, exact date/schedule lookups, or short casual conversation
          (e.g. "hi", "hello", "what time is my meeting?", "what's my schedule?",
           "what is my email?", "cancel my plans", "ok cool", "thanks")

medium  - General knowledge questions, advice, conversational topics, single tool actions, or anything needing a natural language response
          (e.g. "what do you know about me?", "remind me to buy milk", "who are you?",
           "what should I learn first?", "explain transformers", "what's the difference between ML and AI?")

complex - Explicit multi-step planning, in-depth research with synthesis, code generation, or comparative analysis
          (e.g. "research AI trends and write a report", "plan my entire week",
           "build a Python script that does X", "compare these two options in depth")

Output ONLY the single word: simple, medium, or complex.
No explanation. No punctuation. Just the word."""

_VALID_TIERS = {"simple", "medium", "complex"}


class LLMRouter:
    """
    LLM-powered routing engine.

    Call: complexity, category = await router.classify(query, session_id)
    Same interface as FastIntentClassifier.classify() — drop-in replacement.
    """

    def __init__(self, model: str, fallback_classifier: FastIntentClassifier,
                 llm_provider: LLMProvider = None):
        self.model     = model
        self.fallback  = fallback_classifier
        self._llm_provider = llm_provider
        # Per-session cache: {session_id: {query_lower: (complexity, category)}}
        self._cache: Dict[str, Dict[str, Tuple[QueryComplexity, QueryCategory]]] = {}
        self._stats = {"llm_hits": 0, "fallback_hits": 0, "cache_hits": 0}

    async def classify(
        self,
        query: str,
        session_id: str = "default",
    ) -> Tuple[QueryComplexity, QueryCategory]:
        """
        Classify a query using the LLM with regex fallback.

        Returns (QueryComplexity, QueryCategory) — same as FastIntentClassifier.
        Never raises. Guaranteed to return a result.
        """
        query_stripped = query.strip()
        cache_key = query_stripped.lower()

        # ── Cache hit ─────────────────────────────────────────────────────────
        session_cache = self._cache.setdefault(session_id, {})
        if cache_key in session_cache:
            self._stats["cache_hits"] += 1
            return session_cache[cache_key]

        # ── Try LLM classification ────────────────────────────────────────────
        tier = await self._call_llm(query_stripped)

        if tier:
            self._stats["llm_hits"] += 1
            result = self._tier_to_enums(tier, query_stripped)
            logger.debug(
                f"[LLMRouter] '{query_stripped[:50]}' → {tier} (LLM) "
                f"| cache={self._stats['cache_hits']} llm={self._stats['llm_hits']} "
                f"fallback={self._stats['fallback_hits']}"
            )
        else:
            # ── Regex fallback ────────────────────────────────────────────────
            self._stats["fallback_hits"] += 1
            result = self.fallback.classify(query_stripped)
            logger.debug(
                f"[LLMRouter] '{query_stripped[:50]}' → {result[0].value} (REGEX FALLBACK)"
            )

        # Cache and return
        session_cache[cache_key] = result
        return result

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _call_llm(self, query: str) -> Optional[str]:
        """
        Make the tiny LLM classification call.
        Returns "simple" | "medium" | "complex" or None on any failure.
        Hard timeout = ROUTER_LLM_TIMEOUT seconds.
        """
        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user",   "content": query},
        ]
        try:
            if self._llm_provider:
                raw = await asyncio.wait_for(
                    self._llm_provider.generate(
                        messages,
                        temperature=0.0,
                        max_tokens=3,
                        timeout=ROUTER_LLM_TIMEOUT,
                    ),
                    timeout=ROUTER_LLM_TIMEOUT + 1,
                )
            else:
                # Legacy direct ollama fallback
                client = ollama.AsyncClient()
                response = await asyncio.wait_for(
                    client.chat(
                        model=self.model,
                        messages=messages,
                        options={
                            "num_ctx":     ROUTER_NUM_CTX,
                            "temperature": 0.0,
                            "num_predict": 3,
                        },
                    ),
                    timeout=ROUTER_LLM_TIMEOUT,
                )
                raw = response["message"]["content"].strip().lower()
            return self._parse_tier(raw.strip().lower())

        except asyncio.TimeoutError:
            # Timeout is expected when Ollama is slow — this is normal fallback behaviour
            logger.debug(
                f"[LLMRouter] Router LLM timed out ({ROUTER_LLM_TIMEOUT}s) — using regex fallback"
            )
            return None
        except Exception as e:
            logger.debug(f"[LLMRouter] Router LLM failed ({e}) — using regex fallback")
            return None

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tier(raw: str) -> Optional[str]:
        """
        Extract a valid tier word from the LLM output.
        Handles edge cases where LLM adds punctuation or extra words.
        e.g. "simple." → "simple", "I think medium." → "medium"
        """
        # Direct match first (ideal case)
        clean = raw.strip(" .\n\r\t\"'").lower()
        if clean in _VALID_TIERS:
            return clean

        # Scan for any valid tier word anywhere in the response
        for tier in _VALID_TIERS:
            if re.search(rf"\b{tier}\b", raw, re.IGNORECASE):
                return tier

        return None

    @staticmethod
    def _tier_to_enums(
        tier: str,
        query: str,
    ) -> Tuple[QueryComplexity, QueryCategory]:
        """
        Map a tier string to (QueryComplexity, QueryCategory).
        Category is set to a sensible default — the specific handler
        in SimpleHandler/MediumHandler/Planner does the real work.
        """
        if tier == "simple":
            q = query.lower().strip()
            # Greetings
            if re.match(r"^(?:hi+|hello|hey+|good\s+(?:morning|evening|afternoon|night)|howdy|sup|yo)", q):
                return (QueryComplexity.SIMPLE, QueryCategory.GREETING)
            # Progress / status checks
            if re.search(r"\b(?:what(?:'re| are) you doing|status|progress)\b", q):
                return (QueryComplexity.SIMPLE, QueryCategory.PROGRESS_QUERY)
            # Calendar / schedule / plans lookups
            if re.search(
                r"\b(?:event|events|meeting|meetings|appointment|schedule|calendar"
                r"|plan|plans|today|tomorrow|tmrw|tmr|this week|next week|this weekend"
                r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
                r"|morning|evening|tonight|upcoming|agenda)\b",
                q
            ):
                return (QueryComplexity.SIMPLE, QueryCategory.CALENDAR_QUERY)
            # Strict single-field fact lookups only
            if re.match(
                r"^(?:what(?:'?s| is) my|tell me my|show me my) "
                r"(?:phone(?: number)?|email(?: address)?|address|birthday|age|name)\??$",
                q, re.IGNORECASE
            ):
                return (QueryComplexity.SIMPLE, QueryCategory.FACT_RECALL)
            # Everything else the LLM called 'simple' is short casual conversation.
            # Return GENERAL_CHAT so SimpleHandler gives a direct ack without any LLM call.
            return (QueryComplexity.SIMPLE, QueryCategory.GENERAL_CHAT)

        if tier == "medium":
            return (QueryComplexity.MEDIUM, QueryCategory.SINGLE_SEARCH)

        # complex
        return (QueryComplexity.COMPLEX, QueryCategory.MULTI_STEP_RESEARCH)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def clear_cache(self, session_id: Optional[str] = None):
        if session_id:
            self._cache.pop(session_id, None)
        else:
            self._cache.clear()
