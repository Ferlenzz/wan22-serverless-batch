import os, json, time, fcntl
from pathlib import Path
import runpod
from engine import ENGINE

RP_VOLUME = Path(os.environ.get("RP_VOLUME","/runpod-volume"))
QUEUE, LOCKF, RESULTS = RP_VOLUME/"queue"/"jobs.jsonl", RP_VOLUME/"queue"/"lock", RP_VOLUME/"results"
MAX_BATCH = int(os.environ.get("BATCH_MAX_SIZE","20"))
LingerS   = float(os.environ.get("BATCH_LINGER_SEC","5"))

def _append_jsonl(p:Path,o): p.parent.mkdir(parents=True,exist_ok=True); open(p,"a").write(json.dumps(o,ensure_ascii=False)+"\n")
def _read_all(p:Path):
    if not p.exists(): return []
    return [json.loads(x) for x in open(p) if x.strip()]
def _write_all(p:Path,items):
    tmp=p.with_suffix(".tmp")
    with open(tmp,"w") as f:
        for it in items: f.write(json.dumps(it,ensure_ascii=False)+"\n")
    tmp.replace(p)
def _save_result(jid:str,payload:dict):
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/f"{jid}.json").write_text(json.dumps(payload,ensure_ascii=False))
def _get_result(jid:str):
    f=RESULTS/f"{jid}.json"
    return json.loads(f.read_text()) if f.exists() else {"status":"PENDING"}

def _try_lead_and_process():
    LOCKF.parent.mkdir(parents=True,exist_ok=True)
    with open(LOCKF,"w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return {"leader":"no"}

        t0=time.time()
        while time.time()-t0<LingerS: time.sleep(0.05)

        jobs=_read_all(QUEUE); todo,rest=[],[]
        for j in jobs:
            if not j.get("_taken") and len(todo)<MAX_BATCH:
                j["_taken"]=True; todo.append(j)
            else:
                rest.append(j)
        _write_all(QUEUE, rest+todo)

        for task in todo:
            try:
                out=ENGINE.generate_one(task)
            except Exception as e:
                out={"job_id":task["job_id"],"status":"FAILED","error":str(e)}
            _save_result(task["job_id"], out)

        fcntl.flock(lf, fcntl.LOCK_UN)
        return {"leader":"yes","processed":len(todo)}

def handler(event):
    req = event.get("input",{}) or {};  action=req.get("action","enq")
    if action=="enq":
        job={
          "job_id": req.get("job_id") or str(int(time.time()*1000)),
          "user_id": req.get("user_id","anon"),
          "prompt": req["prompt"],
          "image_base64": req.get("image_base64"),
          "image_path":   req.get("image_path"),
          "image_url":    req.get("image_url"),
          "last_image_base64": req.get("last_image_base64"),
          "last_image_path":   req.get("last_image_path"),
          "reuse_last_image":  bool(req.get("reuse_last_image", False)),
          "width": int(req.get("width",480)), "height": int(req.get("height",832)),
          "length": int(req.get("length",64)), "steps": int(req.get("steps",8)),
          "cfg": float(req.get("cfg",2.0)), "lora_pairs": req.get("lora_pairs", []),
          "ts": time.time()
        }
        _append_jsonl(QUEUE, job)
        info=_try_lead_and_process()
        return {"job_id":job["job_id"], "leader":info}

    if action=="result": return _get_result(req["job_id"])
    if action=="health": return {"ok":True}
    return {"error":"unknown action"}

runpod.serverless.start({"handler": handler})
