"""
Learning Engine
================
Persists user behavioural patterns across sessions with confidence scoring.

Backed by a NEW `learned_patterns` table in the EXISTING SQLite database.
Schema migration is safe and idempotent (same pattern as `reminder_sent`
column migration in db_manager.py).

Confidence range: 0.0 → 1.0
  - 0.5  initial / neutral
  - +0.1 per acceptance (max 1.0)
  - 0.3  reset on correction (new value starts fresh)
  - >0.8 → applied automatically without asking
  - >0.5 → surfaced as a suggestion

Persists: pattern_type, context, key, value, confidence, acceptance_count,
          correction_count, last_seen, created_at
"""

import uuid
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# Confidence thresholds
CONFIDENCE_AUTO_APPLY = 0.8   # Applied automatically without asking
CONFIDENCE_SUGGEST    = 0.5   # Surfaced as a proactive suggestion
CONFIDENCE_ACCEPTANCE_BOOST = 0.1
CONFIDENCE_CORRECTION_RESET = 0.3


@dataclass
class LearnedPattern:
    pattern_id:       str
    pattern_type:     str
    context:          str
    key:              str
    value:            str
    confidence:       float
    acceptance_count: int
    correction_count: int
    last_seen:        Optional[str]
    created_at:       str


class LearningEngine:
    """
    Confidence-scored pattern storage backed by SQLite.
    All methods are async-compatible (they are synchronous inside but
    wrapped with await-able wrappers for consistent calling from async code).
    """

    TABLE = "learned_patterns"

    def __init__(self, db_manager):
        self.db = db_manager
        self._ensure_schema()

    # ──────────────────────────────────────────────────────────────────────
    # Schema migration (idempotent)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_schema(self):
        """
        Creates the learned_patterns table if it doesn't exist.
        Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
        """
        conn = self.db.get_connection()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id               TEXT PRIMARY KEY,
                pattern_type     TEXT NOT NULL,
                context          TEXT,
                key              TEXT NOT NULL,
                value            TEXT NOT NULL,
                confidence       REAL DEFAULT 0.5,
                acceptance_count INTEGER DEFAULT 0,
                correction_count INTEGER DEFAULT 0,
                last_seen        TEXT,
                created_at       TEXT NOT NULL
            )
        """)
        # Add index for fast lookups
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_lp_type_ctx_key
            ON {self.TABLE} (pattern_type, context, key)
        """)
        conn.commit()
        logger.debug("[Learning] Schema ready")

    # ──────────────────────────────────────────────────────────────────────
    # Writing patterns
    # ──────────────────────────────────────────────────────────────────────

    async def record_acceptance(
        self,
        pattern_type: str,
        context: str,
        key: str,
        value: str,
    ):
        """
        User accepted/confirmed a behaviour → boost confidence.
        If pattern exists with same value, increment confidence.
        If no pattern exists, create it at 0.5.
        """
        existing = self._load_pattern(pattern_type, context, key)
        now = datetime.now().isoformat()

        if existing and existing.value == value:
            new_confidence = min(1.0, existing.confidence + CONFIDENCE_ACCEPTANCE_BOOST)
            self._update_confidence(
                existing.pattern_id,
                new_confidence,
                acceptance_count=existing.acceptance_count + 1,
                last_seen=now,
            )
            logger.debug(
                f"[Learning] Accepted: {key}={value} "
                f"(confidence {existing.confidence:.2f} → {new_confidence:.2f})"
            )
        else:
            self._create_pattern(pattern_type, context, key, value, confidence=0.5)

    async def record_correction(
        self,
        pattern_type: str,
        context: str,
        key: str,
        old_value: str,
        new_value: str,
    ):
        """
        User corrected a behaviour → penalise old pattern, create/strengthen new.
        Old value confidence resets to 0.3 (needs to prove itself again).
        New value starts at 0.3 (fresh — needs acceptance).
        """
        existing_old = self._load_pattern_by_value(pattern_type, context, key, old_value)
        now = datetime.now().isoformat()

        if existing_old:
            self._update_confidence(
                existing_old.pattern_id,
                confidence=0.3,
                correction_count=existing_old.correction_count + 1,
                last_seen=now,
            )
            logger.debug(
                f"[Learning] Penalised: {key}={old_value} → confidence 0.3"
            )

        # Create or strengthen the corrected pattern
        existing_new = self._load_pattern_by_value(pattern_type, context, key, new_value)
        if existing_new:
            new_confidence = min(1.0, existing_new.confidence + CONFIDENCE_ACCEPTANCE_BOOST)
            self._update_confidence(
                existing_new.pattern_id,
                new_confidence,
                acceptance_count=existing_new.acceptance_count + 1,
                last_seen=now,
            )
        else:
            self._create_pattern(pattern_type, context, key, new_value, confidence=CONFIDENCE_CORRECTION_RESET)

        logger.info(
            f"[Learning] Corrected: {key} '{old_value}' → '{new_value}'"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Reading patterns
    # ──────────────────────────────────────────────────────────────────────

    async def get_pattern(
        self, pattern_type: str, context: str, key: str
    ) -> Optional[LearnedPattern]:
        """Return the highest-confidence pattern for this type/context/key."""
        return self._load_pattern(pattern_type, context, key)

    async def get_suggestions(self, context: str) -> List[LearnedPattern]:
        """
        Return all patterns for this context above the suggestion threshold.
        Used by MemoryAwareExecutor to build proactive hints.
        """
        conn = self.db.get_connection()
        rows = conn.execute(f"""
            SELECT * FROM {self.TABLE}
            WHERE context LIKE ?
              AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT 10
        """, (f"%{context[:40]}%", CONFIDENCE_SUGGEST)).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    async def should_auto_apply(self, pattern_type: str, context: str, key: str) -> Optional[str]:
        """
        If a pattern has confidence >= AUTO_APPLY threshold, return its value.
        The engine uses this to silently apply known preferences without asking.
        """
        pattern = self._load_pattern(pattern_type, context, key)
        if pattern and pattern.confidence >= CONFIDENCE_AUTO_APPLY:
            return pattern.value
        return None

    def get_all_patterns(self, limit: int = 50) -> List[LearnedPattern]:
        """Debugging helper — returns all stored patterns."""
        conn = self.db.get_connection()
        rows = conn.execute(
            f"SELECT * FROM {self.TABLE} ORDER BY confidence DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    # ──────────────────────────────────────────────────────────────────────
    # Internal DB helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_pattern(self, pattern_type: str, context: str, key: str) -> Optional[LearnedPattern]:
        conn = self.db.get_connection()
        row = conn.execute(f"""
            SELECT * FROM {self.TABLE}
            WHERE pattern_type = ? AND context = ? AND key = ?
            ORDER BY confidence DESC
            LIMIT 1
        """, (pattern_type, context, key)).fetchone()
        return self._row_to_pattern(row) if row else None

    def _load_pattern_by_value(
        self, pattern_type: str, context: str, key: str, value: str
    ) -> Optional[LearnedPattern]:
        conn = self.db.get_connection()
        row = conn.execute(f"""
            SELECT * FROM {self.TABLE}
            WHERE pattern_type = ? AND context = ? AND key = ? AND value = ?
            LIMIT 1
        """, (pattern_type, context, key, value)).fetchone()
        return self._row_to_pattern(row) if row else None

    def _create_pattern(
        self,
        pattern_type: str,
        context: str,
        key: str,
        value: str,
        confidence: float = 0.5,
    ):
        conn = self.db.get_connection()
        now = datetime.now().isoformat()
        conn.execute(f"""
            INSERT INTO {self.TABLE}
              (id, pattern_type, context, key, value, confidence, acceptance_count,
               correction_count, last_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
        """, (str(uuid.uuid4()), pattern_type, context, key, value, confidence, now, now))
        conn.commit()

    def _update_confidence(
        self,
        pattern_id: str,
        confidence: float,
        acceptance_count: Optional[int] = None,
        correction_count: Optional[int] = None,
        last_seen: Optional[str] = None,
    ):
        conn = self.db.get_connection()
        sets = ["confidence = ?"]
        params: list = [round(confidence, 4)]
        if acceptance_count is not None:
            sets.append("acceptance_count = ?")
            params.append(acceptance_count)
        if correction_count is not None:
            sets.append("correction_count = ?")
            params.append(correction_count)
        if last_seen:
            sets.append("last_seen = ?")
            params.append(last_seen)
        params.append(pattern_id)
        conn.execute(f"UPDATE {self.TABLE} SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()

    @staticmethod
    def _row_to_pattern(row) -> Optional[LearnedPattern]:
        if not row:
            return None
        return LearnedPattern(
            pattern_id=row["id"],
            pattern_type=row["pattern_type"],
            context=row["context"] or "",
            key=row["key"],
            value=row["value"],
            confidence=float(row["confidence"]),
            acceptance_count=int(row["acceptance_count"]),
            correction_count=int(row["correction_count"]),
            last_seen=row["last_seen"],
            created_at=row["created_at"],
        )
