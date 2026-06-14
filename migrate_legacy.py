"""
Legacy JSON -> SQLite Migration
=================================
Migrates all flat-file data from the old intelligent-memory architecture
into the new friday SQLite database (memory.db).

Migrates:
  1. user_profile.json    -> user_profile table (facts + preferences)
  2. facts.json           -> facts table (episodic/calendar events)
  3. short_term_recall.json -> short_term_recall table
  4. sessions/*.jsonl     -> sessions table
  5. memory/*.txt/md files -> chunks table (for search indexing)
"""

import os
import sys
import json
import uuid
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from friday.memory.db_manager import MemoryDatabaseManager

DB_PATH = "workspace/memory.db"
WORKSPACE = "workspace"
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
DREAMS_DIR = os.path.join(MEMORY_DIR, ".dreams")
SESSIONS_DIR = os.path.join(MEMORY_DIR, "sessions")

db = MemoryDatabaseManager(DB_PATH)
db.ensure_schema()
db.ensure_vector_table(dimensions=768)
conn = db.get_connection()

migrated = {
    "profile_facts": 0,
    "profile_prefs": 0,
    "facts": 0,
    "recall_entries": 0,
    "session_messages": 0,
    "memory_chunks": 0,
    "skipped_duplicates": 0,
}

print("=" * 55)
print("  FRIDAY LEGACY MIGRATION")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# 1. USER PROFILE (user_profile.json)
# ─────────────────────────────────────────────────────────────
profile_path = os.path.join(DREAMS_DIR, "user_profile.json")
if os.path.exists(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    for key, val in profile.get("facts", {}).items():
        conn.execute(
            "INSERT OR REPLACE INTO user_profile (key, category, value) VALUES (?, ?, ?)",
            (key, "facts", str(val))
        )
        migrated["profile_facts"] += 1

    for key, val in profile.get("preferences", {}).items():
        conn.execute(
            "INSERT OR REPLACE INTO user_profile (key, category, value) VALUES (?, ?, ?)",
            (key, "preferences", str(val))
        )
        migrated["profile_prefs"] += 1

    conn.commit()
    print(f"[OK] user_profile.json -> {migrated['profile_facts']} facts, {migrated['profile_prefs']} prefs")
else:
    print(f"[--] No user_profile.json at {profile_path}")

# ─────────────────────────────────────────────────────────────
# 2. EPISODIC FACTS (facts.json)
# ─────────────────────────────────────────────────────────────
facts_path = os.path.join(MEMORY_DIR, "facts.json")
if os.path.exists(facts_path):
    with open(facts_path, "r", encoding="utf-8") as f:
        facts_raw = json.load(f)
    # Handle both list format and dict format: {"facts": {id: {...}, ...}}
    if isinstance(facts_raw, list):
        facts_list = facts_raw
    elif isinstance(facts_raw, dict) and "facts" in facts_raw:
        # Dict-of-dicts: key is the id
        facts_list = []
        for fid, fobj in facts_raw["facts"].items():
            fobj["id"] = fid
            facts_list.append(fobj)
    else:
        facts_list = list(facts_raw.values()) if isinstance(facts_raw, dict) else []

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    for fact in facts_list:
        fid = fact.get("id") or str(uuid.uuid4())
        content = fact.get("content", "")
        date_start = fact.get("date_start", now_iso)
        date_end = fact.get("date_end", date_start)
        importance = fact.get("importance", 0.5)
        confidence = fact.get("confidence", 1.0)
        status = fact.get("status", "active")
        created_at = fact.get("created_at", now_iso)
        reminder_sent = fact.get("reminder_sent", None)

        conn.execute(
            """INSERT OR REPLACE INTO facts 
               (id, content, date_start, date_end, importance, confidence, status, created_at, reminder_sent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fid, content, date_start, date_end, importance, confidence, status, created_at, reminder_sent)
        )
        migrated["facts"] += 1

    conn.commit()
    print(f"[OK] facts.json -> {migrated['facts']} episodic facts")
else:
    print(f"[--] No facts.json at {facts_path}")

# ─────────────────────────────────────────────────────────────
# 3. SHORT-TERM RECALL (short_term_recall.json)
# ─────────────────────────────────────────────────────────────
recall_path = os.path.join(DREAMS_DIR, "short_term_recall.json")
if os.path.exists(recall_path):
    with open(recall_path, "r", encoding="utf-8") as f:
        recall_raw = json.load(f)
    # Format is: {"entries": {key: {...}}, "updatedAt": ..., "version": ...}
    entries_data = recall_raw.get("entries", recall_raw) if isinstance(recall_raw, dict) else {}
    if isinstance(entries_data, list):
        entries = entries_data
    elif isinstance(entries_data, dict):
        entries = list(entries_data.values())
    else:
        entries = []

    for entry in entries:
        key = entry.get("key", "")
        if not key:
            continue

        # Check for existing entry
        existing = conn.execute("SELECT key FROM short_term_recall WHERE key=?", (key,)).fetchone()
        if existing:
            migrated["skipped_duplicates"] += 1
            continue

        conn.execute(
            """INSERT OR IGNORE INTO short_term_recall 
               (key, path, start_line, end_line, source, snippet,
                recall_count, daily_count, grounded_count,
                total_score, max_score,
                first_recalled_at, last_recalled_at,
                query_hashes, recall_days, concept_tags, claim_hash, promoted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                entry.get("path", ""),
                entry.get("start_line", entry.get("startLine", 0)),
                entry.get("end_line", entry.get("endLine", 0)),
                entry.get("source", "memory"),
                entry.get("snippet", "")[:500],
                entry.get("recall_count", entry.get("recallCount", 0)),
                entry.get("daily_count", entry.get("dailyCount", 0)),
                entry.get("grounded_count", entry.get("groundedCount", 0)),
                entry.get("total_score", entry.get("totalScore", 0.0)),
                entry.get("max_score", entry.get("maxScore", 0.0)),
                entry.get("first_recalled_at", entry.get("firstRecalledAt", "")),
                entry.get("last_recalled_at", entry.get("lastRecalledAt", "")),
                json.dumps(entry.get("query_hashes", entry.get("queryHashes", []))),
                json.dumps(entry.get("recall_days", entry.get("recallDays", []))),
                json.dumps(entry.get("concept_tags", entry.get("conceptTags", []))),
                entry.get("claim_hash", entry.get("claimHash", "")),
                entry.get("promoted_at", entry.get("promotedAt", None)),
            )
        )
        migrated["recall_entries"] += 1

    conn.commit()
    print(f"[OK] short_term_recall.json -> {migrated['recall_entries']} entries ({migrated['skipped_duplicates']} skipped)")
else:
    print(f"[--] No short_term_recall.json at {recall_path}")

# ─────────────────────────────────────────────────────────────
# 4. SESSIONS (*.jsonl)
# ─────────────────────────────────────────────────────────────
if os.path.exists(SESSIONS_DIR):
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".jsonl"):
            continue
        session_id = fname.replace(".jsonl", "")
        fpath = os.path.join(SESSIONS_DIR, fname)

        # Check if already migrated
        existing_count = conn.execute(
            "SELECT COUNT(*) as c FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()["c"]
        if existing_count > 0:
            print(f"[--] Session '{session_id}' already migrated ({existing_count} msgs)")
            continue

        session_msgs = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    tool_calls = json.dumps(msg.get("tool_calls", None)) if msg.get("tool_calls") else None
                    created_at = msg.get("created_at", msg.get("timestamp", ""))

                    conn.execute(
                        """INSERT INTO sessions (session_id, role, content, tool_calls, created_at) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (session_id, role, content, tool_calls, created_at or None)
                    )
                    session_msgs += 1
                except json.JSONDecodeError:
                    continue

        conn.commit()
        migrated["session_messages"] += session_msgs
        print(f"[OK] Session '{session_id}' -> {session_msgs} messages")

# ─────────────────────────────────────────────────────────────
# 5. MEMORY FILES -> chunks table
# ─────────────────────────────────────────────────────────────
if os.path.exists(MEMORY_DIR):
    for fname in os.listdir(MEMORY_DIR):
        fpath = os.path.join(MEMORY_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.startswith(".") or fname.endswith(".json") or fname.endswith(".jsonl"):
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:
            continue

        if not content:
            continue

        # Check for duplicate content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        dup = conn.execute(
            "SELECT value FROM meta WHERE key=?", (f"content_hash:{content_hash}",)
        ).fetchone()
        if dup:
            migrated["skipped_duplicates"] += 1
            continue

        chunk_id = str(uuid.uuid4())
        try:
            conn.execute(
                "INSERT INTO chunks (id, path, source, chunkIndex, content) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, fpath, "memory", 0, content[:500])
            )
            try:
                conn.execute(
                    "INSERT INTO chunks_fts (id, path, source, content) VALUES (?, ?, ?, ?)",
                    (chunk_id, fpath, "memory", content[:500])
                )
            except Exception:
                pass
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                (f"content_hash:{content_hash}", fname)
            )
            conn.commit()
            migrated["memory_chunks"] += 1
        except Exception as e:
            print(f"  [WARN] Failed to index {fname}: {e}")

    print(f"[OK] memory/ files -> {migrated['memory_chunks']} chunks indexed")

# ─────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  MIGRATION SUMMARY")
print("=" * 55)
print(f"  Profile facts:    {migrated['profile_facts']}")
print(f"  Profile prefs:    {migrated['profile_prefs']}")
print(f"  Episodic facts:   {migrated['facts']}")
print(f"  Recall entries:   {migrated['recall_entries']}")
print(f"  Session messages: {migrated['session_messages']}")
print(f"  Memory chunks:    {migrated['memory_chunks']}")
print(f"  Skipped (dups):   {migrated['skipped_duplicates']}")
print()

# Re-verify
print("  DATABASE COUNTS AFTER MIGRATION:")
for table in ["user_profile", "facts", "short_term_recall", "sessions", "chunks"]:
    count = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
    print(f"    {table}: {count}")

db.close()
print()
print("  MIGRATION COMPLETE - Friday now has full memory of the past.")
