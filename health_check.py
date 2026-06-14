import sys, os
sys.path.insert(0, '.')
os.chdir(r'c:\Users\prani\OneDrive\Desktop\MY PPROJECTS\friday_mark2')

from friday.memory.db_manager import MemoryDatabaseManager
from friday.memory.layers.layer_6_profile import UserPersonalization
from friday.memory.layers.layer_3_episodic import FactStore
from friday.memory.layers.promotion import get_promotion_stats

# Use the REAL memory.db
db = MemoryDatabaseManager('workspace/memory.db')
db.ensure_schema()
db.ensure_vector_table(dimensions=768)
conn = db.get_connection()

print('=== REAL MEMORY.DB HEALTH CHECK ===')
print()

# 1. Check table counts
for table in ['chunks', 'facts', 'sessions', 'short_term_recall', 'user_profile']:
    try:
        count = conn.execute(f'SELECT COUNT(*) as c FROM {table}').fetchone()['c']
        print(f'  {table}: {count} rows')
    except Exception as e:
        print(f'  {table}: ERROR - {e}')

print()

# 2. Profile
p = UserPersonalization(db)
facts = p.profile.get('facts', {})
prefs = p.profile.get('preferences', {})
print(f'=== USER PROFILE ({len(facts)} facts, {len(prefs)} prefs) ===')
for k, v in facts.items():
    print(f'  Fact | {k}: {v}')
for k, v in prefs.items():
    print(f'  Pref | {k}: {v}')
print()

# 3. Active episodic facts
fs = FactStore(db)
active = fs.get_active_facts()
print(f'=== ACTIVE EVENTS ({len(active)}) ===')
for f in active:
    print(f'  {f.get("content", "?")} | {f.get("date_start", "?")}')
print()

# 4. Promotion stats
stats = get_promotion_stats(db)
print('=== PROMOTION ENGINE ===')
print(f'  tracked={stats["total_tracked"]}, promoted={stats["promoted"]}, pending={stats["pending"]}')
print(f'  avg_score={stats["avg_score"]:.3f}, max_score={stats["max_score"]:.3f}')
print()

# 5. Recent sessions
sessions = conn.execute('SELECT DISTINCT session_id FROM sessions ORDER BY id DESC LIMIT 5').fetchall()
print(f'=== RECENT SESSIONS ({len(sessions)}) ===')
for s in sessions:
    sid = s['session_id']
    msgs = conn.execute('SELECT COUNT(*) as c FROM sessions WHERE session_id=?', (sid,)).fetchone()['c']
    last = conn.execute('SELECT content, role FROM sessions WHERE session_id=? ORDER BY id DESC LIMIT 1', (sid,)).fetchone()
    snippet = str(last['content'])[:60] if last else '?'
    print(f'  {sid}: {msgs} msgs | last: [{last["role"] if last else "?"}] {snippet}')
print()

# 6. Sample chunks
print('=== SAMPLE CHUNKS (last 5) ===')
chunks = conn.execute('SELECT id, source, content FROM chunks ORDER BY rowid DESC LIMIT 5').fetchall()
for c in chunks:
    print(f'  [{c["source"]}] {c["content"][:80]}')
print()

# 7. Short-term recall top entries
print('=== TOP SHORT-TERM RECALL ENTRIES ===')
top = conn.execute('SELECT key, recall_count, total_score, promoted_at FROM short_term_recall ORDER BY total_score DESC LIMIT 5').fetchall()
for r in top:
    promoted = 'PROMOTED' if r['promoted_at'] else 'pending'
    print(f'  recalls={r["recall_count"]} score={r["total_score"]:.3f} [{promoted}] {r["key"][:60]}')
if not top:
    print('  (no entries yet)')
print()

db.close()
print('=== HEALTH CHECK COMPLETE ===')
