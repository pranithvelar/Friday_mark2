import sqlite3
db = sqlite3.connect('workspace/memory.db')
db.row_factory = sqlite3.Row
rows = db.execute('SELECT id, content, status, date_start, date_end FROM facts ORDER BY date_start').fetchall()
print(f'ALL {len(rows)} facts in DB:')
for r in rows:
    status = r["status"]
    content = r["content"][:70]
    start = r["date_start"]
    end = r["date_end"]
    print(f'  [{status:10}] {repr(content)}')
    print(f'               start={start} end={end}')
db.close()
