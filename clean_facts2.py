"""
Final pass: 
- Fix the math test duplicate (keep only the version with proper time, delete the other)
- Mark earning goal as 'active' (it was correctly stored, just got incorrectly contested)
- Uncontested the earning goal since it doesn't overlap with a time-specific event
"""
import sqlite3

db = sqlite3.connect('workspace/memory.db')
db.row_factory = sqlite3.Row

# Math test: we have two rows for the same event
tests = db.execute(
    "SELECT id, content, date_start, date_end, status FROM facts WHERE content LIKE '%math algebra%' AND status != 'deleted'"
).fetchall()
print("Math test rows:")
for r in tests:
    print(f"  id={r['id'][:8]} start={r['date_start']} end={r['date_end']}")

# Keep the one with date_end=2026-06-17T10:00:00 (proper time block), delete the other
if len(tests) == 2:
    # find which has the tighter end time (the 08:00-10:00 one is the real one)
    keep_id = None
    delete_id = None
    for r in tests:
        if '08:00' in r['date_start'] or '10:00' in r['date_end']:
            keep_id = r['id']
        else:
            delete_id = r['id']
    if keep_id and delete_id:
        db.execute("UPDATE facts SET status='deleted' WHERE id=?", (delete_id,))
        db.execute("UPDATE facts SET status='active' WHERE id=?", (keep_id,))
        print(f"  Kept: {keep_id[:8]}, Deleted: {delete_id[:8]}")
    else:
        # just keep the first, delete the second
        db.execute("UPDATE facts SET status='deleted' WHERE id=?", (tests[1]['id'],))
        db.execute("UPDATE facts SET status='active' WHERE id=?", (tests[0]['id'],))
        print(f"  Kept: {tests[0]['id'][:8]}, Deleted: {tests[1]['id'][:8]}")

# Earning goal: mark as active (no time overlap with math test since it spans a year)
db.execute(
    "UPDATE facts SET status='active' WHERE content LIKE '%21 lakhs%' AND status='contested'"
)
print("Earning goal: set to active")

db.commit()

# Final state
rows = db.execute(
    "SELECT id, content, status, date_start, date_end FROM facts WHERE status IN ('active','contested') ORDER BY date_start"
).fetchall()
print(f"\nFinal clean state: {len(rows)} facts")
for r in rows:
    print(f"  [{r['status']:10}] {r['content'][:70]!r}")
    print(f"               {r['date_start']} → {r['date_end']}")

db.close()
