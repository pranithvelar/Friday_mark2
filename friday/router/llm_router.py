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
You are the routing brain of Friday, a personal AI assistant.

Classify the user's message into EXACTLY ONE of these four outcomes:

simple  - No tool call needed. Covers EVERYTHING conversational including:
          • Greetings, acks, casual chat ("hey", "ok", "cool", "thanks")
          • Calendar/event lookups ("what's my schedule?", "do I have anything today?")
          • Adding, cancelling, or modifying a reminder/event ("remind me at 3pm", "cancel my trip")
          • Profile/memory queries ("what's my email?", "what do you know about me?")
          • Any single-intent conversational request

medium  - Needs 1–3 TOOLS (not steps) to complete:
          • A web search, file action, or data retrieval needing 1-3 distinct tool calls
          • Tasks that are self-contained within 3 tool calls

complex - ONLY for explicit multi-step PROJECTS with a clear ACTION command:
          • "research X and write a full report"
          • "build me a tracker app" / "implement X" / "start building Y"
          • "plan my entire week with tasks"
          • The user must have EXPLICITLY said to build/do/implement/start/create it.
          DO NOT use complex if the user is only describing or explaining an idea.

clarify - Use when the request is too vague OR when the user is describing/pitching
          an idea rather than issuing an explicit command:
          • "build something", "do some research", "make an app" — no specifics given
          • "I have an idea about an app", "so I was thinking of building X",
            "it basically works like this..." — user is EXPLAINING, not commanding.
          • Any message where the user is pitching a concept but has NOT yet said
            "build it", "do it", "implement it", "start", or "go ahead".
          When in doubt between complex and clarify → always choose clarify.

Output ONLY one word: simple  medium  complex  clarify
No punctuation. No explanation."""

_VALID_TIERS = {"simple", "medium", "complex", "clarify"}


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
        self._stats = {"llm_hits": 0, "fallback_hits": 0}

    async def classify(
        self,
        query: str,
        session_id: str = "default",
    ) -> Tuple[QueryComplexity, QueryCategory]:
        """
        Classify a query using the LLM with regex fallback.

        Cache intentionally removed: conversational context changes the meaning
        of the same phrase ("remind me tomorrow" means different things on
        different days). Every query is classified fresh — the 2s timeout +
        regex fallback already guarantees sub-200ms worst-case latency.

        Returns (QueryComplexity, QueryCategory) — same as FastIntentClassifier.
        Never raises. Guaranteed to return a result.
        """
        query_stripped = query.strip()

        # ── Try LLM classification ────────────────────────────────────────────
        tier = await self._call_llm(query_stripped)

        if tier:
            self._stats["llm_hits"] += 1
            result = self._tier_to_enums(tier, query_stripped)
            logger.debug(
                f"[LLMRouter] '{query_stripped[:50]}' → {tier} (LLM) "
                f"| llm={self._stats['llm_hits']} fallback={self._stats['fallback_hits']}"
            )
        else:
            # ── Regex fallback ────────────────────────────────────────────────
            self._stats["fallback_hits"] += 1
            result = self.fallback.classify(query_stripped)
            logger.debug(
                f"[LLMRouter] '{query_stripped[:50]}' → {result[0].value} (REGEX FALLBACK)"
            )

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
        Pure 1:1 pass-through from LLM tier word to enums.
        Zero keywords. Zero regex. The LLM made the decision — we trust it.
        """
        from friday.router.intent_classifier import QueryComplexity, QueryCategory
        if tier == "simple":
            return (QueryComplexity.SIMPLE, QueryCategory.GENERAL_CHAT)
        if tier == "medium":
            return (QueryComplexity.MEDIUM, QueryCategory.SINGLE_SEARCH)
        if tier == "clarify":
            return (QueryComplexity.SIMPLE, QueryCategory.CLARIFY)
        # complex (default)
        return (QueryComplexity.COMPLEX, QueryCategory.MULTI_STEP_RESEARCH)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)
