# handler.py
import os, json, time, fcntl, base64, uuid, re
from pathlib import Path
import runpod
from engine import ENGINE

# -----------------------------
# Paths / env
# -----------------------------
RP_VOLUME = Path(os.environ.get("RP_VOLUME", "/runpod-volume"))
QUEUE_DIR   = RP_VOLUME / "queue"
RESULTS_DIR = RP_VOLUME / "results"
STATE_DIR   = RP_VOLUME / "state"
LAST_DIR    = STATE_DIR / "last_images"

QUEUE_FILE  = QUEUE_DIR / "jobs.jsonl"
LOCK_FILE   = QUEUE_DIR / "lock"

MAX_BATCH   = int(os.environ.get("BATCH_MAX_SIZE", "20"))
LINGER_SEC  = float(os.environ.get("BATCH_LINGER_SEC", "5"))

for p in (QUEUE_DIR, RESULTS_DIR, LAST_DIR):
    p.mkdir(parents=True, exist_ok=True)

_name_re = re.compile(r"[^a-zA-Z0-9._-]+")


def _norm_user_id(uid: str) -> str:
    uid = uid.strip()
    uid = _name_re.sub("_", uid or "u1")
    return uid[:64]


def _append_jsonl(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_all(p: Path):
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_all(p: Path, items: list):
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _save_json_result(job_id: str, payload: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{job_id}.json").write_text(json.dumps(payload, ensure_ascii=False))


def _load_json_result(job_id: str) -> dict:
    f = RESULTS_DIR / f"{job_id}.json"
    return json.loads(f.read_text()) if f.exists() else {"status": "PENDING"}


def _try_lead_and_process():
    """
    Лидер пытается за N секунд собрать пачку и прогнать ENGINE одним вызовом.
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"leader": "no"}

        t0 = time.time()
        while True:
            jobs = _read_all(QUEUE_FILE)
            if not jobs:
                return {"leader": "idle"}

            if len(jobs) >= MAX_BATCH or (time.time() - t0) >= LINGER_SEC:
                batch = jobs[:MAX_BATCH]
                rest  = jobs[MAX_BATCH:]
                _write_all(QUEUE_FILE, rest)

                # Запускаем модель на пачке
                res = ENGINE.process_batch(batch, RESULTS_DIR, LAST_DIR)
                # res — список по порядку batch с полями {job_id, status, video_path, ...}
                # JSON-файлы уже записаны в ENGINE, здесь просто возвращаем отчёт
                return {"leader": "ran", "batch": len(batch)}
            else:
                time.sleep(0.05)  # короткий backoff


def _save_last_image(last_image_b64: str, user_id: str) -> Path | None:
    """
    last_image_b64 может приходить напрямую с клиента (если он хочет «зафиксировать» стейт).
    В любом случае ENGINE тоже кладёт last frame → мы не мешаем.
    """
    if not last_image_b64:
        return None
    try:
        user = _norm_user_id(user_id)
        LAST_DIR.mkdir(parents=True, exist_ok=True)
        outp = LAST_DIR / f"{user}.png"
        # поддержка dataURL
        if "," in last_image_b64:
            last_image_b64 = last_image_b64.split(",", 1)[1]
        raw = base64.b64decode(last_image_b64)
        outp.write_bytes(raw)
        return outp
    except Exception:
        return None


def handler(event):
    """
    Единая точка входа serverless.
    """
    # 0) попробуем стать лидером и прогнать пачку «между делом»
    _try_lead_and_process()

    inp = event.get("input") or {}
    action = (inp.get("action") or "").lower()

    if action in ("health", "ping", "status"):
        return {"ok": True}

    # Синхронная генерация (runsync): отдать видео прямо в ответе
    if action in ("generate_sync", "runsync"):
        user_id = _norm_user_id(str(inp.get("user_id") or "u1"))
        _save_last_image(inp.get("last_image_base64") or "", user_id)

        job = {
            "job_id": "sync-" + str(uuid.uuid4()),
            "user_id": user_id,
            "prompt": str(inp.get("prompt") or ""),
            "width": int(inp.get("width") or 480),
            "height": int(inp.get("height") or 832),
            "length": int(inp.get("length") or 81),
            "steps": int(inp.get("steps") or 10),
            "cfg": float(inp.get("cfg") or 2.0),
            "image_base64": inp.get("image_base64"),
            "lora_pairs": inp.get("lora_pairs") or [],
        }

        result = ENGINE.process_batch([job], RESULTS_DIR, LAST_DIR)[0]
        # Возвращаем Base64, чтобы тебе не нужно было лезть на том, но также файл уже сохранён
        payload = {
            "status": result["status"],
            "job_id": result["job_id"],
            "video_path": result.get("video_path"),
            "video": result.get("video_base64"),  # <= можно выключить для экономии трафика
        }
        _save_json_result(job["job_id"], payload)
        return payload

    # Async-режим: ставим в очередь
    if action in ("enqueue", "enq", "generate"):
        user_id = _norm_user_id(str(inp.get("user_id") or "u1"))
        _save_last_image(inp.get("last_image_base64") or "", user_id)

        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "user_id": user_id,
            "prompt": str(inp.get("prompt") or ""),
            "width": int(inp.get("width") or 480),
            "height": int(inp.get("height") or 832),
            "length": int(inp.get("length") or 81),
            "steps": int(inp.get("steps") or 10),
            "cfg": float(inp.get("cfg") or 2.0),
            "image_base64": inp.get("image_base64"),
            "lora_pairs": inp.get("lora_pairs") or [],
        }
        _append_jsonl(QUEUE_FILE, job)
        return {"ack": True, "job_id": job_id, "queue_size": len(_read_all(QUEUE_FILE))}

    # Получить результат async-задачи
    if action in ("get", "result", "fetch"):
        job_id = str(inp.get("job_id") or "")
        return _load_json_result(job_id)

    return {"error": f"unknown action: {action}"}


runpod.serverless.start({"handler": handler})
