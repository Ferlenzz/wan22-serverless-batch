import os, time
from loguru import logger
import runpod

from engine import generate_one

RATE_USD_PER_SEC = float(os.environ.get('COST_RATE_USD_PER_SEC', '0.00031'))  # rough estimate

def handler(event):
    """RunPod Serverless handler.
    Expected event['input']:
      action: 'health' | 'run'
      prompt, image_base64, width, height, steps, cfg, length
    """
    inp = (event or {}).get('input') or {}
    action = (inp.get('action') or 'run').lower()

    if action == 'health':
        return {'ok': True, 'status': 'ready'}

    # default path: run generation
    t0 = time.time()
    res = generate_one(inp)
    exec_secs = time.time() - t0

    if not res.get('ok'):
        return {'ok': False, 'error': res.get('error', 'unknown')}

    cost = exec_secs * RATE_USD_PER_SEC
    out = {
        'ok': True,
        'video': res['video_b64'],
        'seconds': res['seconds'],
        'saved_path': res['path'],
        'estimated_cost_usd': round(cost, 6),
    }
    logger.info("Done in {}s, saved {}", res['seconds'], res['path'])
    return out

if __name__ == "__main__":
    logger.info("Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
