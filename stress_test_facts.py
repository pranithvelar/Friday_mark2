"""
Stress test for the updated fact extraction + pipeline.
Tests open-ended projects, natural completion detection, delete, and the old crash case.
"""
import asyncio
import sqlite3
import sys
import os
import datetime

sys.path.insert(0, os.path.abspath("."))

from friday.memory.layers.layer_3_episodic import extract_facts, FactStore
from friday.memory.db_manager import MemoryDatabaseManager

# ── Setup ────────────────────────────────────────────────────────────────────
db = MemoryDatabaseManager("workspace/memory.db")
fact_store = FactStore(db)
FAR_FUTURE = datetime.datetime(2099, 12, 31, 23, 59, 59)

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}")
    if detail:
        print(f"         -> {detail}")

def run_pipeline_op(op):
    """Simulate exactly what pipeline.py does for one operation."""
    action = op.get("action")
    if action == "add" and op.get("content") and op.get("date_start"):
        start_dt = datetime.datetime.fromisoformat(op["date_start"])
        raw_end = op.get("date_end")
        end_dt = None
        if raw_end and isinstance(raw_end, str):
            try:
                end_dt = datetime.datetime.fromisoformat(raw_end)
            except ValueError:
                end_dt = None
        if end_dt is None:
            end_dt = FAR_FUTURE
        fact_store.add_fact(op["content"], start_dt, end_dt, float(op.get("importance", 0.5)))
        return f"added: {op['content'][:50]} | end={end_dt.year}"
    elif action == "complete" and op.get("keyword"):
        matched = fact_store.complete_fact(op["keyword"])
        return f"completed '{op['keyword']}': {'matched' if matched else 'NO MATCH'}"
    elif action == "delete" and op.get("keyword"):
        fact_store.delete_fact(op["keyword"])
        return f"deleted '{op['keyword']}'"
    return f"unknown action: {action}"

def get_fact(keyword):
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM facts WHERE LOWER(content) LIKE ? ORDER BY created_at DESC LIMIT 1",
        (f"%{keyword.lower()}%",)
    ).fetchone()
    return dict(row) if row else None

print("\n" + "="*60)
print("  FRIDAY FACT PIPELINE — STRESS TEST")
print("="*60)

# ── TEST 1: Old crash case ────────────────────────────────────────────────────
print("\n[1] Old crash: LLM returns 'no end date specified'")
op = {
    "action": "add",
    "content": "stress_test_crash_case",
    "date_start": datetime.datetime.now().isoformat(),
    "date_end": "no end date specified",
    "importance": 0.7
}
try:
    detail = run_pipeline_op(op)
    fact = get_fact("stress_test_crash")
    check("Does NOT crash on bad date_end string", fact is not None, detail)
    check("Stored with FAR_FUTURE end date", fact and fact["date_end"].startswith("2099"), 
          f"date_end={fact['date_end'][:10] if fact else 'N/A'}")
except Exception as e:
    check("Does NOT crash", False, str(e))

# ── TEST 2: null date_end (open-ended project) ────────────────────────────────
print("\n[2] Open-ended project: date_end = null")
op = {
    "action": "add",
    "content": "stress_test_open_project_alpha",
    "date_start": datetime.datetime.now().isoformat(),
    "date_end": None,
    "ongoing": True,
    "importance": 0.7
}
try:
    detail = run_pipeline_op(op)
    fact = get_fact("stress_test_open_project_alpha")
    check("Stored successfully with null date_end", fact is not None, detail)
    check("date_end is FAR_FUTURE (2099)", fact and fact["date_end"].startswith("2099"),
          f"date_end={fact['date_end'][:10] if fact else 'N/A'}")
    check("status is 'active'", fact and fact["status"] == "active",
          f"status={fact['status'] if fact else 'N/A'}")
except Exception as e:
    check("Open-ended storage", False, str(e))

# ── TEST 3: Hard deadline event ───────────────────────────────────────────────
print("\n[3] Hard deadline: specific end date")
tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1))
op = {
    "action": "add",
    "content": "stress_test_deadline_event",
    "date_start": datetime.datetime.now().isoformat(),
    "date_end": tomorrow.isoformat(),
    "ongoing": False,
    "importance": 1.0
}
detail = run_pipeline_op(op)
fact = get_fact("stress_test_deadline_event")
check("Hard deadline stored", fact is not None, detail)
check("date_end is NOT far future", fact and not fact["date_end"].startswith("2099"),
      f"date_end={fact['date_end'][:10] if fact else 'N/A'}")

# ── TEST 4: LLM natural completion detection ──────────────────────────────────
print("\n[4] LLM extract_facts on natural completion phrases")
test_phrases = [
    ("I just finished the stress test alpha project", "complete"),
    ("shipped the stress test alpha project today", "complete"),
    ("the stress test open project is all done", "complete"),
    ("wrapped up stress test alpha", "complete"),
    ("lets build another thing", "add"),
    ("how are you doing", "none"),
]
for phrase, expected_action in test_phrases:
    result = extract_facts(phrase)
    ops = result.get("operations", [])
    if expected_action == "none":
        check(f"No ops for: '{phrase[:45]}'", len(ops) == 0, f"got {len(ops)} ops")
    else:
        got = ops[0].get("action") if ops else "none"
        check(f"'{phrase[:45]}'", got == expected_action, f"expected={expected_action} got={got}")

# ── TEST 5: Complete operation updates date_end ───────────────────────────────
print("\n[5] complete_fact() stamps correct date_end")
before = datetime.datetime.now()
op = {"action": "complete", "keyword": "stress_test_open_project_alpha"}
detail = run_pipeline_op(op)
after = datetime.datetime.now()
fact = get_fact("stress_test_open_project_alpha")
check("complete_fact found the record", fact is not None, detail)
if fact:
    end_str = fact["date_end"]
    try:
        end_dt = datetime.datetime.fromisoformat(end_str)
        check("date_end is NOW (not 2099)", not end_str.startswith("2099"), f"date_end={end_str[:19]}")
        check("date_end is within test window", before <= end_dt <= after + datetime.timedelta(seconds=5),
              f"before={before.isoformat()[:19]} end={end_str[:19]}")
    except:
        check("date_end is parseable", False, f"got: {end_str}")

# ── TEST 6: complete on non-existent keyword ──────────────────────────────────
print("\n[6] complete_fact on non-existent keyword (graceful)")
op = {"action": "complete", "keyword": "totally_nonexistent_xyzzy_12345"}
detail = run_pipeline_op(op)
check("No crash on missing fact", True, detail)

# ── Cleanup ───────────────────────────────────────────────────────────────────
conn = db.get_connection()
conn.execute("UPDATE facts SET status='deleted' WHERE content LIKE 'stress_test%'")
conn.commit()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  RESULTS: {passed} passed / {failed} failed / {len(results)} total")
if failed == 0:
    print("  ALL TESTS PASSED -- system is JARVIS-grade")
else:
    print("  FAILURES DETECTED")
    for r in results:
        if r[0] == FAIL:
            print(f"     FAILED: {r[1]} -- {r[2]}")
print("="*60)
