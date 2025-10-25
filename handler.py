import os, json, time, fcntl, base64, uuid, re
from pathlib import Path
import runpod
from loguru import logger
from engine import ENGINE
from queue import enqueue, dequeue_batch, mark_done

RP_VOLUME   = Path(os.environ.get("RP_VOLUME", "/runpod-volume"))
STATE_DIR   = RP_VOLUME / "state"
LAST_DIR    = STATE_DIR / "last_images"
RESULTS_DIR = RP_VOLUME / "results"

STATE_DIR.mkdir(parents=True, exist_ok=True)
LAST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_MAX   = int(os.environ.get("BATCH_MAX_SIZE", "20"))
LINGER_SEC  = float(os.environ.get("BATCH_LINGER_SEC", "5"))

_name_re = re.compile(r"[^a-zA-Z0-9._-]+")

def _norm_user_id(uid: str) -> str:
    uid = (uid or "u1").strip()
    uid = _name_re.sub("_", uid)
    return uid[:64]

def _result_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"

def _save_json(job_id: str, payload: dict):
    (_result_path(job_id)).write_text(json.dumps(payload, ensure_ascii=False))

def _load_json(job_id: str) -> dict:
    f = _result_path(job_id)
    if f.exists():
        return json.loads(f.read_text())
    return {"status": "PENDING"}

def _make_dataurl_from_png_bytes(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()

# ------------------ batch thread (leader) ------------------
def _batch_once():
    """Снимаем пачку из sqlite-очереди и обрабатываем."""
    batch = dequeue_batch(BATCH_MAX)
    if not batch:
        return 0
    t0 = time.time()

    # подстановка last image при необходимости
    for it in batch:
        uid = it.get("user_id","u1")
        reuse = bool(it.get("use_last") or False)
        if reuse and not it.get("image_base64"):
            p = LAST_DIR / f"{uid}.png"
            if p.exists():
                it["image_base64"] = _make_dataurl_from_png_bytes(p.read_bytes())

    results = ENGINE.process_batch(batch, RESULTS_DIR, LAST_DIR)
    exec_ms = int((time.time()-t0)*1000)

    for r in results:
        res_payload = {
            "status": r.get("status","COMPLETED"),
            "job_id": r["job_id"],
            "video_path": r.get("video_path"),
            # "video": r.get("video_base64"),  # если хочешь хранить Base64 — включи
            "exec_ms": r.get("exec_ms", exec_ms)
        }
        _save_json(r["job_id"], res_payload)
        mark_done(r["job_id"])

    logger.info(f"BATCH: {len(batch)} items, exec={exec_ms}ms")
    return len(batch)

def _batch_loop():
    logger.info(f"Batch loop started: max={BATCH_MAX}, linger={LINGER_SEC}s")
    while True:
        try:
            time.sleep(LINGER_SEC)
            _batch_once()
        except Exception as e:
            logger.exception(f"batch error: {e}")

# Запускаем фон
import threading
threading.Thread(target=_batch_loop, daemon=True).start()

# ------------------ handler ------------------
def handler(event):
    inp = event.get("input") or {}
    action = str(inp.get("action") or "health").lower()

    if action in ("health","ping","status"):
        return {"ok": True}

    # синхронный путь (для замеров и отладки): возвращает видео в ответе
    if action in ("runsync","direct","generate_sync"):
        uid = _norm_user_id(inp.get("user_id","u1"))
        # reuse_last
        if inp.get("use_last") and not inp.get("image_base64"):
            p = LAST_DIR / f"{uid}.png"
            if p.exists():
                inp["image_base64"] = _make_dataurl_from_png_bytes(p.read_bytes())

        t0 = time.time()
        job = {
            "job_id": "sync-" + str(uuid.uuid4()),
            "user_id": uid,
            "prompt": str(inp.get("prompt") or ""),
            "width": int(inp.get("width",480)),
            "height": int(inp.get("height",832)),
            "length": int(inp.get("length",64)),
            "steps": int(inp.get("steps",8)),
            "cfg": float(inp.get("cfg",2.0)),
            "image_base64": inp.get("image_base64"),
            "lora_pairs": inp.get("lora_pairs") or [],
        }
        res = ENGINE.process_batch([job], RESULTS_DIR, LAST_DIR)[0]
        exec_ms = int((time.time()-t0)*1000)

        # отдаем base64 и/или путь на томе
        payload = {
            "status": res.get("status","COMPLETED"),
            "job_id": res["job_id"],
            "video_path": res.get("video_path"),
            "video": res.get("video_base64"),  # можно убрать для экономии сети
            "exec_ms": exec_ms
        }
        _save_json(job["job_id"], payload)
        return payload

    # асинхронный путь: ставим в очередь (дешёвая генерация)
    if action in ("enqueue","enq","generate"):
        uid = _norm_user_id(inp.get("user_id","u1"))
        job_id = str(uuid.uuid4())
        payload = {
            "job_id": job_id,
            "user_id": uid,
            "prompt": str(inp.get("prompt") or ""),
            "width": int(inp.get("width",480)),
            "height": int(inp.get("height",832)),
            "length": int(inp.get("length",64)),
            "steps": int(inp.get("steps",8)),
            "cfg": float(inp.get("cfg",2.0)),
            "image_base64": inp.get("image_base64"),
            "lora_pairs": inp.get("lora_pairs") or [],
            "use_last": bool(inp.get("use_last", False)),
        }
        enqueue(job_id, uid, payload)
        # маленький «tick» лидеру, вдруг свободен
        _ = _batch_once()
        return {"ack": True, "job_id": job_id}

    # получить результат async-задачи
    if action in ("get","result","fetch"):
        job_id = str(inp.get("job_id") or "")
        return _load_json(job_id)

    return {"error": f"unknown action: {action}"}

runpod.serverless.start({"handler": handler})
