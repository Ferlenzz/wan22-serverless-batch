import os, time
from loguru import logger
import runpod

from engine import generate_one, warmup

RATE_USD_PER_SEC = float(os.environ.get('COST_RATE_USD_PER_SEC', '0.00031'))  # ~ $0.31/hr

def handler(event):
    payload = event.get("input") or {}
    action = (payload.get("action") or "run").lower()

    if action in ("health", "status"):
        return {"ok": True, "status": "ready"}

    if action == "warmup":
        return {"ok": True, **warmup()}

    t0 = time.time()
    try:
        res = generate_one(payload)
    except Exception as e:
        logger.exception("Generation failed")
        return {"ok": False, "error": str(e)}

    sec = float(res.get("seconds", time.time() - t0))
    cost = round(sec * RATE_USD_PER_SEC, 6)
    return {
        "ok": True,
        "video": res["video_b64"],
        "seconds": sec,
        "saved_path": res["path"],
        "last_image_path": res.get("last_image_path"),
        "estimated_cost_usd": cost,
    }

if __name__ == "__main__":
    logger.info("Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
