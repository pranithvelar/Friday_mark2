"""
Tier 1: Fast Intent Classifier
================================
Pure pattern-matching classifier. NO LLM involved.
Target latency: <50ms per classification.

Routing priority (mirrors OpenClaw's tiered binding priority):
  SIMPLE  → Direct memory/DB lookup. No tools needed.
  MEDIUM  → 1–3 tool calls. AgentLoop with max_steps=3.
  COMPLEX → Multi-step, multi-tool. Full MultiAgentPlanner.
"""

import re
from enum import Enum
from typing import Tuple, List


class QueryComplexity(Enum):
    SIMPLE = "simple"       # <200ms, direct DB/facts
    MEDIUM = "medium"       # <2s, max 3 tool calls
    COMPLEX = "complex"     # >2s, multi-agent planner


class QueryCategory(Enum):
    # ── SIMPLE ──────────────────────────────────────────────
    GREETING        = "greeting"         # "Hello", "Hey Friday"
    CALENDAR_QUERY  = "calendar_query"   # "What's my meeting today?"
    FACT_RECALL     = "fact_recall"      # "What's my favourite colour?"
    TEMPORAL_FACT   = "temporal_fact"    # "When is my flight?"
    PROGRESS_QUERY  = "progress_query"   # "What are you doing?" / "What's the status?"
    GENERAL_CHAT    = "general_chat"     # Short casual chat: "ok", "cancel plans", "thanks", "got it"

    # ── MEDIUM ──────────────────────────────────────────────
    SINGLE_SEARCH   = "single_search"    # "Search for X"
    SINGLE_ACTION   = "single_action"    # "Add reminder / set alarm"
    DATA_EXTRACTION = "data_extraction"  # "Get key points from this"

    # ── COMPLEX ─────────────────────────────────────────────
    MULTI_STEP_RESEARCH = "multi_step_research"  # "Research X and summarise"
    PLANNING_TASK       = "planning_task"        # "Plan my week / create roadmap"
    CODE_GENERATION     = "code_generation"      # "Build / write / implement"
    ANALYSIS_TASK       = "analysis_task"        # "Analyse X and give me insights"


# ---------------------------------------------------------------------------
# Pattern registry
# Each entry: (QueryCategory, [raw_pattern, ...])
# Order within a complexity tier does NOT matter — first match wins per tier.
# ---------------------------------------------------------------------------
_PATTERN_REGISTRY: List[Tuple[QueryComplexity, QueryCategory, List[str]]] = [

    # ── SIMPLE ──────────────────────────────────────────────────────────────
    (QueryComplexity.SIMPLE, QueryCategory.PROGRESS_QUERY, [
        r"(?i)\bwhat(?:'?re| are) you (?:doing|working on|up to)\b",
        r"(?i)\b(?:what'?s|whats) (?:the )?(?:status|progress|update)\b",
        r"(?i)\bhow far (?:are|have) (?:you|we)\b",
        r"(?i)\b(?:current(?:ly)?|right now) doing\b",
        r"(?i)\bwhere are (?:you|we) (?:at|up to)\b",
    ]),

    (QueryComplexity.SIMPLE, QueryCategory.GREETING, [
        r"(?i)^(?:hi|hello|hey|good\s+(?:morning|evening|afternoon|night)|sup|yo|howdy)[\s!?.]*$",
        r"(?i)^(?:hi|hello|hey)\s+\w+[\s!?.]*$",  # catches "Hello Friday", "Hey Jarvis" etc.
        r"(?i)^(?:what'?s up|how are you|how'?s it going)[\s?!]*$",
    ]),

    (QueryComplexity.SIMPLE, QueryCategory.CALENDAR_QUERY, [
        r"(?i)(?:what(?:'?s| is)|whats|when|show|list|do i have).+(?:event|meeting|appointment|schedule|calendar|plan|plans)",
        r"(?i)(?:next|this|upcoming|any).+(?:week|day|month|weekend)",
        r"(?i)(?:do i have|any(?:thing)?|whats?\s+on).+(?:today|tomorrow|tmrw|tmr|this week|tonight)",
        r"(?i)(?:what are|show me|list|any|tell me).{0,10}(?:my )?(?:plan|plans|schedule|agenda|events?)\b",
        r"(?i)(?:any plans?|got plans?).+(?:today|tomorrow|tmrw|tmr|this week|tonight|weekend)",
        r"(?i)(?:remind|reminder|when is|what time).+(?:event|meeting|appointment)",
    ]),

    (QueryComplexity.SIMPLE, QueryCategory.FACT_RECALL, [
        # ONLY match ultra-specific single-field lookups — NOT open-ended questions
        r"(?i)^what(?:'?s| is) my (?:phone(?: number)?|email(?: address)?|address|birthday|age|favourite colou?r|favorite colou?r|favourite food|favorite food|name)\??$",
        r"(?i)^(?:tell me |show me )?my (?:phone(?: number)?|email(?: address)?|address|birthday)\??$",
    ]),


    (QueryComplexity.SIMPLE, QueryCategory.TEMPORAL_FACT, [
        r"(?i)(?:when is|when was|what time).+(?:flight|trip|holiday|vacation|deadline|exam|interview)",
        r"(?i)(?:how many (?:days|hours|weeks)).+(?:until|till|to|before)",
        r"(?i)(?:what'?s|whats).+(?:due|deadline|eta|date)",
    ]),

    # ── MEDIUM ──────────────────────────────────────────────────────────────
    (QueryComplexity.MEDIUM, QueryCategory.SINGLE_SEARCH, [
        r"(?i)^(?:search|find|look up|google|lookup|look for)\b",
        r"(?i)(?:what is|who is|where is|how does)\s+(?!my|i |we )[\w\s]{3,}",
        r"(?i)(?:find|get|fetch|retrieve).+(?:information|info|data|details)\s+(?:about|on|for)\b",
        r"(?i)(?:check|look up).+(?:price|stock|weather|news|latest)",
    ]),

    (QueryComplexity.MEDIUM, QueryCategory.SINGLE_ACTION, [
        r"(?i)^(?:add|create|set|schedule|remind me|delete|remove|update|mark)\b",
        r"(?i)(?:remind me|set an? (?:alarm|reminder|timer)).+(?:at|in|on|for)",
        r"(?i)(?:add|put).+(?:to|in|on).+(?:calendar|list|reminder|note)",
        r"(?i)(?:delete|remove|cancel).+(?:event|reminder|note|entry)",
    ]),

    (QueryComplexity.MEDIUM, QueryCategory.DATA_EXTRACTION, [
        r"(?i)(?:extract|pull out|summarise|summarize).+(?:key points|highlights|main|from)\b",
        r"(?i)(?:what are the|list the|give me the).+(?:key|main|important).+(?:points|ideas|things)\b",
    ]),

    # ── COMPLEX ─────────────────────────────────────────────────────────────
    (QueryComplexity.COMPLEX, QueryCategory.MULTI_STEP_RESEARCH, [
        r"(?i)(?:research|investigate|deep.?dive|explore).+(?:and|then|also)",
        r"(?i)(?:compare|contrast).+(?:and|vs\.?|versus)",
        r"(?i)(?:pros? and cons?|advantages? and disadvantages?|for and against)",
        r"(?i)(?:comprehensive|thorough|detailed|in-depth).+(?:report|analysis|overview|summary)",
        r"(?i)research\s+(?:about|on)\s+\w",
    ]),

    (QueryComplexity.COMPLEX, QueryCategory.PLANNING_TASK, [
        r"(?i)(?:plan|organise|organize|create a plan|help me plan|map out)\b",
        r"(?i)(?:step.?by.?step|breakdown|roadmap|strategy|workflow)\b",
        r"(?i)(?:what (?:should|do) i do.+(?:order|first|next|then))",
        r"(?i)(?:create|build|design|draft).+(?:plan|schedule|roadmap|strategy|workflow)",
    ]),

    (QueryComplexity.COMPLEX, QueryCategory.CODE_GENERATION, [
        r"(?i)(?:write|create|build|develop|code|implement|generate).+(?:script|program|app(?:lication)?|function|class|module|api|bot)",
        r"(?i)(?:show me(?: the)? code|give me(?: the)? code|write (?:me )?(?:a |the )?code)\b",
        r"(?i)(?:make|build) (?:me )?(?:a |an )?(?:working|simple|full|complete)\b",
    ]),

    (QueryComplexity.COMPLEX, QueryCategory.ANALYSIS_TASK, [
        r"(?i)(?:analyse|analyze).+(?:and|then).+(?:give|provide|tell)",
        r"(?i)(?:what (?:does|do|did) .+ (?:mean|indicate|suggest|show|imply))",
        r"(?i)(?:find (?:patterns?|trends?|insights?|correlations?|anomalies?))\b",
        r"(?i)(?:evaluate|assess|review).+(?:and|then).+(?:give|suggest|recommend)",
    ]),
]


class FastIntentClassifier:
    """
    Precompiles all patterns at init time. Thread-safe for read-only classify() calls.
    classify() is O(n_patterns) but n is small (~50) and regex ops are fast.
    """

    def __init__(self):
        # Compile all patterns once at construction time
        self._compiled: List[Tuple[QueryComplexity, QueryCategory, List[re.Pattern]]] = []
        for complexity, category, raw_patterns in _PATTERN_REGISTRY:
            compiled_pats = [re.compile(p) for p in raw_patterns]
            self._compiled.append((complexity, category, compiled_pats))

    def classify(self, query: str) -> Tuple[QueryComplexity, QueryCategory]:
        """
        Classify a user query into (QueryComplexity, QueryCategory).

        Priority order:
          1. SIMPLE categories (check first — most common, fastest exit)
          2. MEDIUM categories
          3. COMPLEX categories
          4. Heuristic fallback (short query → SIMPLE, else → MEDIUM)
        """
        query = query.strip()

        # Check SIMPLE first
        for complexity, category, patterns in self._compiled:
            if complexity != QueryComplexity.SIMPLE:
                continue
            for pat in patterns:
                if pat.search(query):
                    return (QueryComplexity.SIMPLE, category)

        # Check MEDIUM
        for complexity, category, patterns in self._compiled:
            if complexity != QueryComplexity.MEDIUM:
                continue
            for pat in patterns:
                if pat.search(query):
                    return (QueryComplexity.MEDIUM, category)

        # Check COMPLEX
        for complexity, category, patterns in self._compiled:
            if complexity != QueryComplexity.COMPLEX:
                continue
            for pat in patterns:
                if pat.search(query):
                    return (QueryComplexity.COMPLEX, category)

        # ── Heuristic fallback ───────────────────────────────────────────────
        # ALL unknown queries go to MEDIUM so the LLM answers naturally.
        # Never silently dump raw data for an unknown query.
        return (QueryComplexity.MEDIUM, QueryCategory.SINGLE_SEARCH)

    def _matches_any(self, query: str, patterns: List[re.Pattern]) -> bool:
        for pat in patterns:
            if pat.search(query):
                return True
        return False
