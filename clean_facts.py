"""
Clean the Friday DB:
1. Delete all garbage 'active' facts that are clearly conversation turns (not real events)
2. Keep only real events: math test, earning goal, ECE studies (but clean duplicates)
"""
import sqlite3
import datetime

db = sqlite3.connect('workspace/memory.db')
db.row_factory = sqlite3.Row

# First show what we're working with
rows = db.execute("SELECT id, content, status, date_start, date_end FROM facts").fetchall()
print(f"Before: {len(rows)} total facts")

# Garbage patterns: conversation text stored as "events"
GARBAGE_CONTENT = [
    'description',
    'what did u plan',
    'Understood, Sir. On it.',
    'call me sir nothing else',
    '3rd year in ECE studies and some events lined up for later',
    'Starting 3rd year in ECE studies',   # duplicate from today's chat session (not a real event)
]

# Delete ALL facts whose content exactly matches these garbage phrases
for phrase in GARBAGE_CONTENT:
    result = db.execute(
        "UPDATE facts SET status='deleted' WHERE content=? AND status!='deleted'",
        (phrase,)
    )
    if result.rowcount:
        print(f"  Deleted {result.rowcount} garbage fact(s): {phrase!r}")

# Delete duplicates of the math test and schedule call (keep the cleanest one)
# Mark the timezone-confused duplicates as deleted
# "Plan to earn 21 lakhs" - keep the contested one with end=2027, delete the rest
db.execute("""
    UPDATE facts SET status='deleted'
    WHERE content LIKE '%21 lakhs%' 
    AND id NOT IN (
        SELECT id FROM facts WHERE content LIKE '%21 lakhs%' AND status='contested'
        ORDER BY date_start ASC LIMIT 1
    )
    AND status != 'deleted'
""")
print(f"  Cleaned duplicate '21 lakhs' facts")

# Mark the math test that has end=2099 as deleted (bad LLM extraction, no real end date given)
db.execute("""
    UPDATE facts SET status='deleted'
    WHERE content LIKE '%math algebra test%'
    AND date_end LIKE '2099%'
    AND status != 'deleted'
""")
print(f"  Cleaned open-ended math test duplicate (date_end=2099)")

# "Schedule call" - delete all (these were hallucinated from bot responses)
db.execute("""
    UPDATE facts SET status='deleted'
    WHERE content LIKE '%Schedule call%'
    AND status != 'deleted'
""")
print(f"  Deleted all 'Schedule call' facts (hallucinated from bot output)")

db.commit()

# Final state
rows = db.execute("SELECT id, content, status, date_start FROM facts WHERE status IN ('active','contested') ORDER BY date_start").fetchall()
print(f"\nAfter cleanup: {len(rows)} active/contested facts remaining:")
for r in rows:
    print(f"  [{r['status']:10}] {r['content'][:70]!r}  ({r['date_start'][:10]})")

db.close()
print("\nDone.")
