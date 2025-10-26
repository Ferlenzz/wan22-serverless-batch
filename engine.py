import os
import io
import time
import base64
from typing import Dict, Any, Optional
from pathlib import Path

# ensure WAN repo is importable
import sys
WAN_ROOT = os.environ.get("WAN_ROOT", "/app/Wan2.2")
if os.path.isdir(WAN_ROOT) and WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

from loguru import logger
from PIL import Image
import numpy as np
import imageio

# ------------------ WAN 2.2 ------------------
# cfg is an EasyDict object (do NOT call it)
from wan.configs.wan_ti2v_5B import ti2v_5B as cfg
from wan.image2video import WanI2V

# paths
RP_VOLUME = Path(os.environ.get("RP_VOLUME", "/runpod-volume"))
CKPT_DIR = Path(os.environ.get("WAN_CKPT_DIR", RP_VOLUME / "models" / "Wan2.2-TI2V-5B"))
RESULTS_DIR = RP_VOLUME / "results"
LAST_IMG_DIR = RP_VOLUME / "last_images"
for d in [RESULTS_DIR, LAST_IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_MODEL: Optional[WanI2V] = None

def _ensure_cfg():
    # Some old revs expose cfg as callable. Normalize to object.
    return cfg() if callable(cfg) else cfg

def _load_model() -> WanI2V:
    global _MODEL
    if _MODEL is None:
        c = _ensure_cfg()
        logger.info(f"Loading WanI2V from checkpoints dir: {CKPT_DIR}")
        _MODEL = WanI2V(c, str(CKPT_DIR), device_id=0)
        logger.info("WanI2V ready.")
    return _MODEL

def _b64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _image_to_b64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _save_last_image(user_id: str, im: Image.Image) -> str:
    p = LAST_IMG_DIR / f"{user_id}.png"
    im.save(p, format="PNG")
    return str(p)

def _load_last_image(user_id: str) -> Optional[Image.Image]:
    p = LAST_IMG_DIR / f"{user_id}.png"
    if p.exists():
        return Image.open(p).convert("RGB")
    return None

def _frames_to_mp4_b64(frames: np.ndarray, fps: int = 24) -> str:
    tmp = RESULTS_DIR / f"tmp_{int(time.time()*1000)}.mp4"
    imageio.mimsave(tmp, frames, fps=fps, macro_block_size=None)
    with open(tmp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return "data:video/mp4;base64," + b64

def warmup() -> Dict[str, Any]:
    _ = _load_model()
    return {"ok": True, "message": "model initialized"}

def generate_one(inp: Dict[str, Any]) -> Dict[str, Any]:
    m = _load_model()
    user_id = str(inp.get("user_id", "u1"))
    use_last = bool(inp.get("use_last", False))
    prompt = str(inp.get("prompt", ""))

    img: Optional[Image.Image] = None
    b64 = (inp.get("image_base64") or "").strip()
    if b64:
        img = _b64_to_image(b64)
        _save_last_image(user_id, img)
    elif use_last:
        img = _load_last_image(user_id)
        if img is None:
            raise RuntimeError("use_last=True but no previous image found; send image_base64 once.")
    else:
        raise RuntimeError("No image provided. Send image_base64 or set use_last=true after previous call.")

    width = int(inp.get("width", 480))
    height = int(inp.get("height", 832))
    steps = int(inp.get("steps", 10))
    cfg_scale = float(inp.get("cfg", 2.0))
    frame_num = int(inp.get("length", 81))

    t0 = time.time()
    frames = m(img=img, prompt=prompt, width=width, height=height,
               steps=steps, guidance_scale=cfg_scale, frame_num=frame_num)
    secs = time.time() - t0

    if isinstance(frames, (list, tuple)):
        arr = np.stack(frames, axis=0)
    else:
        arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[..., None]
    arr = arr.astype(np.uint8)

    out_path = RESULTS_DIR / f"wan22_{int(time.time())}.mp4"
    imageio.mimsave(out_path, arr, fps=24, macro_block_size=None)
    with open(out_path, "rb") as f:
        video_b64 = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

    return {
        "video_b64": video_b64,
        "path": str(out_path),
        "seconds": round(secs, 3),
        "last_image_path": str(LAST_IMG_DIR / f"{user_id}.png"),
    }
