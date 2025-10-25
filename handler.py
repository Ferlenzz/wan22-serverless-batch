import os, runpod, base64
from engine import generate_one, _start_comfy_once

RATE_PER_SEC = float(os.environ.get("RATE_PER_SEC", "0.00031"))  # пример для оценки

def handler(event: dict):
    payload = event.get("input", {}) or {}
    action  = payload.get("action", "gen")

    if action == "health":
        return {"ok": True}

    if action == "warmup":
        _start_comfy_once()
        return {"ok": True}

    # общие поля
    prompt = payload.get("prompt", "cinematic portrait, soft rim light")
    width  = int(payload.get("width" , 480))
    height = int(payload.get("height", 832))
    steps  = int(payload.get("steps" , 8))
    cfg    = float(payload.get("cfg", 2.0))
    length = int(payload.get("length", 81))
    user_id = str(payload.get("user_id", "u1"))
    use_last = bool(payload.get("use_last", True))
    image_base64 = payload.get("image_base64")

    if action == "runsync":
        res = generate_one(prompt, width, height, steps, cfg, length,
                           user_id=user_id, use_last=use_last,
                           image_base64=image_base64, return_base64=True)
        cost = (res.get("exec_ms", 0)/1000.0) * RATE_PER_SEC
        return {"status":"COMPLETED", "output": {"video": res.get("video_base64"),
                                                 "mp4_path": res.get("mp4_path"),
                                                 "est_cost_usd": round(cost, 6)}}

    # action == "gen" — обычный async, без возврата base64 (дешевле по сети)
    res = generate_one(prompt, width, height, steps, cfg, length,
                       user_id=user_id, use_last=use_last, image_base64=image_base64,
                       return_base64=False)
    cost = (res.get("exec_ms", 0)/1000.0) * RATE_PER_SEC
    return {"status":"COMPLETED", "output": {"mp4_path": res.get("mp4_path"),
                                             "est_cost_usd": round(cost, 6)}}

runpod.serverless.start({"handler": handler})
