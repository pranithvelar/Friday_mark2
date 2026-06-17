"""
MemoryDecayWatcher — Background Confidence Decay for Layer 4 (Semantic Memory)
==============================================================================
Registered with BackgroundScheduler. Called every ~60 seconds by the scheduler,
but self-throttles: actual decay math only runs every DECAY_INTERVAL_HOURS hours.

What it decays:
  - semantic_facts: facts that haven't been confirmed recently lose confidence
    slowly using exponential decay (half-life = HALF_LIFE_DAYS days).

    Example (45-day half-life):
      - A fact confirmed once, never seen again:
          → hits 0.5 confidence at 45 days
          → drops below CONFIDENCE_MIN_DISPLAY (0.4) at ~60 days
          → becomes invisible in context injection (but NOT deleted)

What it does NOT decay (immune):
  - source='stated': user explicitly told Friday this fact — permanent until
    the user explicitly corrects or deletes it.
  - evidence >= EVIDENCE_DECAY_IMMUNE: facts reinforced many times are too
    consolidated to passively decay.

What it does NOT touch:
  - Layer 6 (user_profile) — explicit preferences/facts are permanent
  - Layer 2 (short_term_recall) — handled by PromotionEngine.prune_stale_entries()
  - Session messages / calendar facts (facts table) — separate lifecycle

Returns None always (no user-facing alert). Silent background job.
"""

import math
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────
DECAY_INTERVAL_HOURS  = 6      # Self-throttle: only run every 6 hours
HALF_LIFE_DAYS        = 45.0   # Confidence half-life for unconfirmed facts
CONFIDENCE_FLOOR      = 0.1    # Never decay below this — prevents hard-zeroing facts
EVIDENCE_DECAY_IMMUNE = 5      # Facts with >= this many observations are immune
STATED_SOURCE_IMMUNE  = True   # source='stated' facts are permanent (user explicitly stated)
CONFIDENCE_MIN_DISPLAY = 0.4   # Below this, facts become invisible in context (from layer_4)


class MemoryDecayWatcher:
    """
    Background watcher: applies temporal confidence decay to Layer 4 semantic facts.
    Follows the same check(db_manager, config) interface as ProactiveEventsWatcher
    and GoogleWorkspaceWatcher — plugs straight into BackgroundScheduler.

    Usage (in chat.py):
        bg_scheduler.register_watcher(MemoryDecayWatcher())
    """

    def __init__(self):
        self._last_run_ts: float = 0.0   # Unix timestamp of last real decay run

    async def check(self, db_manager, config) -> None:
        """
        Called every ~60 seconds by BackgroundScheduler.run().
        Self-throttles: actual decay math only fires every DECAY_INTERVAL_HOURS hours.
        Always returns None — this is a silent job, never produces a user alert.
        """
        now = time.time()
        elapsed_hours = (now - self._last_run_ts) / 3600.0

        if elapsed_hours < DECAY_INTERVAL_HOURS:
            return None   # Not time yet — fast return, zero work done

        self._last_run_ts = now

        try:
            decayed, dropped = _run_decay_cycle(db_manager)
            if decayed > 0:
                logger.info(
                    f"[MemoryDecay] Decayed {decayed} semantic fact(s) | "
                    f"{dropped} dropped below display threshold (still stored, not deleted)."
                )
            else:
                logger.debug("[MemoryDecay] Decay cycle ran — no facts needed updating.")
        except Exception as e:
            # Never crash the scheduler — decay failure is non-fatal
            logger.warning(f"[MemoryDecay] Decay cycle failed (non-fatal): {e}")

        return None   # Always return None — scheduler checks 'if alert:' before printing


def _run_decay_cycle(db_manager) -> tuple:
    """
    The actual decay logic. Pure SQLite — no LLM calls, no network, fast.
    Runs synchronously (acceptable: typically <5ms, batch SQL update).

    Algorithm:
        new_confidence = max(CONFIDENCE_FLOOR, confidence * exp(-λ * age_days))
        where λ = ln(2) / HALF_LIFE_DAYS

    Returns: (n_decayed, n_dropped_below_display_threshold)
    """
    conn = db_manager.get_connection()
    now_dt  = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    # Load all active fact candidates
    rows = conn.execute(
        """
        SELECT id, confidence, evidence, source, last_confirmed_at
        FROM semantic_facts
        WHERE status = 'active'
        """
    ).fetchall()

    if not rows:
        return 0, 0

    lam = math.log(2) / HALF_LIFE_DAYS   # decay constant for half-life

    n_decayed = 0
    n_dropped = 0
    updates   = []   # batch: (new_confidence, updated_at, fact_id)

    for row in rows:
        fact_id    = row["id"]
        confidence = row["confidence"]
        evidence   = row["evidence"]
        source     = row["source"]
        last_conf  = row["last_confirmed_at"]

        # ── Immunity checks ────────────────────────────────────────────────
        if STATED_SOURCE_IMMUNE and source == "stated":
            continue   # User explicitly stated this — never passively decay

        if evidence >= EVIDENCE_DECAY_IMMUNE:
            continue   # Seen and confirmed many times — too consolidated to decay

        # ── Age calculation ────────────────────────────────────────────────
        if last_conf:
            try:
                last_dt = datetime.fromisoformat(
                    last_conf.replace("Z", "+00:00")
                ).replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now_dt - last_dt).total_seconds() / 86400.0)
            except Exception:
                age_days = 0.0
        else:
            # No confirmation timestamp — fact is brand new, skip decay
            age_days = 0.0

        if age_days <= 0:
            continue   # Confirmed today or no age data — no decay applied

        # ── Exponential decay ──────────────────────────────────────────────
        # Why exponential and not linear?
        #   - Linear: same penalty per day regardless of age — unfair to old facts
        #   - Exponential: mirrors cognitive forgetting curves; recent facts decay
        #     faster when uncertain, slows as the fact stabilises over time
        decay_factor   = math.exp(-lam * age_days)
        new_confidence = max(CONFIDENCE_FLOOR, confidence * decay_factor)

        # Skip writes where the change is negligibly small (< 0.1% shift)
        if abs(new_confidence - confidence) < 0.001:
            continue

        n_decayed += 1
        if new_confidence < CONFIDENCE_MIN_DISPLAY:
            n_dropped += 1   # Will become invisible in context injection

        updates.append((round(new_confidence, 4), now_iso, fact_id))

    # Batch update in a single transaction — efficient, atomic
    if updates:
        conn.executemany(
            "UPDATE semantic_facts SET confidence=?, updated_at=? WHERE id=?",
            updates
        )
        conn.commit()

    return n_decayed, n_dropped
