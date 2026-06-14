import sqlite3
import os
os.chdir(r'c:\Users\prani\OneDrive\Desktop\MY PPROJECTS\friday_mark2')

db = sqlite3.connect('workspace/memory.db')
db.row_factory = sqlite3.Row

tables = [r['name'] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('Tables in memory.db:')
for t in tables:
    try:
        count = db.execute(f"SELECT COUNT(*) as c FROM [{t}]").fetchone()['c']
        print(f'  {t}: {count} rows')
    except Exception as e:
        print(f'  {t}: ERROR {e}')

print()
print("=== USER PROFILE RAW ===")
try:
    rows = db.execute("SELECT key, category, value FROM user_profile LIMIT 20").fetchall()
    for r in rows:
        print(f"  [{r['category']}] {r['key']} = {r['value']}")
    if not rows:
        print("  (empty)")
except Exception as e:
    print(f"  user_profile not found: {e}")

print()
print("=== FACTS RAW ===")
try:
    rows = db.execute("SELECT content, status, date_start FROM facts LIMIT 10").fetchall()
    for r in rows:
        print(f"  [{r['status']}] {r['content']} | {r['date_start']}")
    if not rows:
        print("  (empty)")
except Exception as e:
    print(f"  facts not found: {e}")

print()
print("=== CHUNKS SAMPLE ===")
try:
    rows = db.execute("SELECT id, source, content FROM chunks LIMIT 5").fetchall()
    for r in rows:
        print(f"  [{r['source']}] {r['content'][:80]}")
    if not rows:
        print("  (empty)")
except Exception as e:
    print(f"  chunks not found: {e}")

print()
print("=== SHORT TERM RECALL ===")
try:
    rows = db.execute("SELECT key, recall_count, total_score FROM short_term_recall ORDER BY total_score DESC LIMIT 5").fetchall()
    for r in rows:
        print(f"  recalls={r['recall_count']} score={r['total_score']:.3f} | {r['key'][:60]}")
    if not rows:
        print("  (empty)")
except Exception as e:
    print(f"  short_term_recall not found: {e}")

print()
print("=== SESSIONS ===")
try:
    sessions = db.execute("SELECT DISTINCT session_id FROM sessions").fetchall()
    for s in sessions:
        count = db.execute("SELECT COUNT(*) as c FROM sessions WHERE session_id=?", (s['session_id'],)).fetchone()['c']
        print(f"  {s['session_id']}: {count} messages")
    if not sessions:
        print("  (empty)")
except Exception as e:
    print(f"  sessions not found: {e}")

db.close()
print()
print("=== RAW DB INSPECTION COMPLETE ===")
