import os, sqlite3, json, time
from pathlib import Path
from typing import List, Dict, Any

DB_PATH = Path(os.environ.get("RP_VOLUME","/runpod-volume")) / "state" / "jobs.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY,
        user_id TEXT,
        payload TEXT,
        ts REAL,
        status TEXT
    )""")
    c.commit()
    return c

def enqueue(job_id: str, user_id: str, payload: Dict[str,Any]):
    c = _conn()
    c.execute("INSERT OR REPLACE INTO jobs(id,user_id,payload,ts,status) VALUES (?,?,?,?,?)",
              (job_id, user_id, json.dumps(payload), time.time(), "PENDING"))
    c.commit(); c.close()

def dequeue_batch(limit: int) -> List[Dict[str,Any]]:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT id,user_id,payload FROM jobs WHERE status='PENDING' ORDER BY ts LIMIT ?", (limit,))
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    if ids:
        cur.executemany("UPDATE jobs SET status='TAKEN' WHERE id=?", [(i,) for i in ids])
        c.commit()
    c.close()
    out=[]
    for jid, uid, payload in rows:
        p = json.loads(payload)
        p["job_id"]=jid; p["user_id"]=uid
        out.append(p)
    return out

def mark_done(job_id: str):
    c = _conn()
    c.execute("UPDATE jobs SET status='DONE' WHERE id=?", (job_id,))
    c.commit(); c.close()
