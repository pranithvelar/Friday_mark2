"""
Memory-Aware Executor
======================
Queries memory BEFORE each execution step so the engine makes
decisions informed by YOUR preferences, past patterns, and stored facts.

Connected to REAL project modules:
  - UserPersonalization  (src/memory/personalization.py)
  - HybridSearcher       (src/search/hybrid_search.py)

Public API (called by ExecutionEngine per step):
  pre_step_check(step, session_id) → MemoryCheckResult
  build_proactive_suggestion(step, context) → Optional[str]
  apply_personalization(step) → ExecutionStep (modified copy)
  learn_from_correction(original, corrected, session_id)
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class MemoryCheckResult:
    """Returned by pre_step_check() — structured advice for the engine."""

    def __init__(
        self,
        suggestions:        List[str],
        constraints:        List[str],
        personalized_params: Dict[str, Any],
    ):
        self.suggestions         = suggestions
        self.constraints         = constraints
        self.personalized_params = personalized_params

    def has_advice(self) -> bool:
        return bool(self.suggestions or self.constraints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestions":         self.suggestions,
            "constraints":         self.constraints,
            "personalized_params": self.personalized_params,
        }


class MemoryAwareExecutor:
    """
    Pre-step memory consultant for the ExecutionEngine.

    Instantiated with the real personalization and searcher objects
    already live in terminal_chat.py — no duplication.
    """

    def __init__(self, personalization, searcher=None, learning_engine=None):
        self.personalization = personalization
        self.searcher        = searcher
        self.learning        = learning_engine

        # ── Preference key map: action keywords → user_profile key ────────
        self._PREF_KEY_MAP = {
            "file":    ["file_naming_convention", "file_location", "favourite_folder"],
            "name":    ["naming_convention", "file_naming_convention"],
            "colour":  ["favourite_colour", "preferred_colour"],
            "color":   ["favourite_colour", "preferred_colour"],
            "format":  ["preferred_format", "output_format"],
            "style":   ["writing_style", "response_style"],
            "email":   ["email_tone", "email_style", "email_signature"],
            "folder":  ["favourite_folder", "default_folder"],
            "sheet":   ["sheet_naming", "spreadsheet_naming"],
        }

    # ──────────────────────────────────────────────────────────────────────
    # Main pre-step check
    # ──────────────────────────────────────────────────────────────────────

    async def pre_step_check(self, step, session_id: str) -> MemoryCheckResult:
        """
        Called by ExecutionEngine BEFORE each step.
        Returns personalised suggestions and constraints.

        Example:
          Step action: "Create a new spreadsheet"
          → checks "sheet_naming" preference
          → returns suggestion: "Sir, shall I name it 'Summary_2024-01-15' like your recent files?"
        """
        suggestions = []
        constraints = []
        personalized_params = {}

        action_lower = step.action.lower()

        # 1. Check relevant user preferences
        relevant_prefs = self._get_relevant_preferences(action_lower)
        for pref_key, pref_val in relevant_prefs.items():
            friendly_key = pref_key.replace("_", " ")
            suggestion = self.build_proactive_suggestion(step, pref_key, pref_val)
            if suggestion:
                suggestions.append(suggestion)
            personalized_params[pref_key] = pref_val

        # 2. Check learned patterns (high-confidence only)
        if self.learning:
            patterns = await self.learning.get_suggestions(context=step.action[:60])
            for pattern in patterns:
                if pattern.confidence >= 0.7:
                    suggestions.append(
                        f"Based on your history: prefer {pattern.value} for {pattern.key}"
                    )
                    personalized_params[pattern.key] = pattern.value

        # 3. Check memory search for relevant past behaviour
        if self.searcher and step.tool_category in ("memory", "search"):
            try:
                results = await self.searcher.search(
                    step.action, vector_weight=0.7, text_weight=0.3, max_results=2
                )
                for r in results:
                    if r.snippet:
                        suggestions.append(f"Relevant memory: {r.snippet[:120]}")
            except Exception as e:
                logger.debug(f"[MemoryExec] Memory search failed: {e}")

        return MemoryCheckResult(
            suggestions=suggestions,
            constraints=constraints,
            personalized_params=personalized_params,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Proactive suggestion builder
    # ──────────────────────────────────────────────────────────────────────

    def build_proactive_suggestion(self, step, pref_key: str, pref_val: str) -> Optional[str]:
        """
        Constructs a JARVIS-style proactive suggestion.
        E.g.: "Sir, shall I name it 'Summary_2024-01-15' like your recent files?"
        """
        action_lower = step.action.lower()

        if "file" in pref_key and ("create" in action_lower or "save" in action_lower or "name" in action_lower):
            return f"Sir, shall I use your preferred naming convention '{pref_val}' for this file?"

        if "colour" in pref_key or "color" in pref_key:
            return f"Sir, you prefer {pref_val} — shall I apply that here?"

        if "sheet" in pref_key or "spreadsheet" in pref_key:
            return f"Sir, shall I name the sheet using your convention '{pref_val}'?"

        if "folder" in pref_key:
            return f"Sir, would you like this saved to your preferred folder: '{pref_val}'?"

        if "style" in pref_key or "format" in pref_key:
            return f"Sir, I will apply your preferred {pref_key.replace('_', ' ')}: '{pref_val}'."

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Personalization application
    # ──────────────────────────────────────────────────────────────────────

    def apply_personalization(self, step, personalized_params: Dict[str, Any]):
        """
        Modify step parameters based on known preferences before execution.
        Mutates step in-place (safe — step is only accessed by the engine coroutine).
        """
        # Example: if file naming convention known, inject it into step action
        naming = personalized_params.get("file_naming_convention")
        if naming and ("create" in step.action.lower() or "save" in step.action.lower()):
            if naming not in step.action:
                step.action = f"{step.action} (use naming convention: {naming})"

        colour = personalized_params.get("favourite_colour")
        if colour and ("colour" in step.action.lower() or "color" in step.action.lower()):
            if colour not in step.action:
                step.action = f"{step.action} (use colour: {colour})"

    # ──────────────────────────────────────────────────────────────────────
    # Contextual learning from user corrections
    # ──────────────────────────────────────────────────────────────────────

    async def learn_from_correction(self, original_step, corrected_description: str, session_id: str):
        """
        Called when the user says "don't do it like this, do it like that."
        Delegates to LearningEngine to update the pattern.
        """
        if not self.learning:
            return
        await self.learning.record_correction(
            pattern_type="step_approach",
            context=original_step.action[:60],
            key="preferred_approach",
            old_value=original_step.tool_category,
            new_value=corrected_description[:100],
        )
        logger.info(
            f"[MemoryExec] Learned correction: '{original_step.action[:40]}' → '{corrected_description[:40]}'"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_relevant_preferences(self, action_lower: str) -> Dict[str, str]:
        """
        Scan action text for keywords and look up matching user preferences.
        Returns {pref_key: pref_value} for all matches found.
        """
        if not self.personalization:
            return {}

        matched = {}
        for keyword, pref_keys in self._PREF_KEY_MAP.items():
            if keyword not in action_lower:
                continue
            for pref_key in pref_keys:
                val = (
                    self.personalization.get_preference(pref_key)
                    or self.personalization.get_fact(pref_key)
                )
                if val:
                    matched[pref_key] = str(val)
        return matched
