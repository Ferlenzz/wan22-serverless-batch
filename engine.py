import os
import io
import time
import base64
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# --- чтобы интерпретатор видел /app/Wan2.2 ---
WAN_ROOT = os.environ.get("WAN_ROOT", "/app/Wan2.2")
if os.path.isdir(WAN_ROOT) and WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

from loguru import logger
from PIL import Image
import numpy as np
import imageio

# ------------------ Wan 2.2 ------------------
from wan.configs.wan_ti2v_5B import ti2v_5B as cfg
from wan.image2video import WanI2V

# --- EasyDict совместимость ---
try:
    from easydict import EasyDict as EDict
except Exception:
    class EDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

# ----- пути -----
RP_VOLUME   = Path(os.environ.get("RP_VOLUME", "/runpod-volume"))
CKPT_DIR    = Path(os.environ.get("WAN_CKPT_DIR", RP_VOLUME / "models" / "Wan2.2-TI2V-5B"))
RESULTS_DIR = RP_VOLUME / "results"
LAST_IMG_DIR= RP_VOLUME / "last_images"
for d in (RESULTS_DIR, LAST_IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)

_MODEL: Optional[WanI2V] = None


# ================= helpers =================
def _ensure_cfg():
    """
    Берём конфиг ti2v_5B как объект и добавляем недостающие поля.
    В некоторых ревизиях у config отсутствует `boundary` — добавляем дефолт.
    """
    c = cfg() if callable(cfg) else cfg

    # гарантируем наличие блока boundary
    if not hasattr(c, "boundary") or c.boundary is None:
        c.boundary = EDict({
            "enabled": False,   # отключено по умолчанию
            "lower": 0.0,
            "upper": 1.0,
        })
    else:
        # нормализуем ключ к 'enabled'
        if isinstance(c.boundary, dict):
            if "enable" in c.boundary and "enabled" not in c.boundary:
                c.boundary["enabled"] = bool(c.boundary.get("enable"))
        else:
            # объект с атрибутами
            if hasattr(c.boundary, "enable") and not hasattr(c.boundary, "enabled"):
                try:
                    c.boundary.enabled = bool(getattr(c.boundary, "enable"))
                except Exception:
                    pass

    return c


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


def _save_last_image(user_id: str, im: Image.Image) -> str:
    p = LAST_IMG_DIR / f"{user_id}.png"
    im.save(p, format="PNG")
    return str(p)


def _load_last_image(user_id: str) -> Optional[Image.Image]:
    p = LAST_IMG_DIR / f"{user_id}.png"
    if p.exists():
        return Image.open(p).convert("RGB")
    return None


def _arr_to_mp4_and_b64(arr: np.ndarray, out_path: Path, fps: int = 24) -> str:
    """
    arr: (T,H,W,3) uint8
    сохраняет в out_path и возвращает data:video/mp4;base64,...
    """
    imageio.mimsave(out_path, arr, fps=fps, macro_block_size=None)
    with open(out_path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()


# ================ public API =================
def warmup() -> Dict[str, Any]:
    """
    Ленивая инициализация модели — подтянет веса и прогреет пайплайн.
    """
    _ = _load_model()
    return {"ok": True, "message": "model initialized"}


def generate_one(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ожидает в inp:
      - prompt: str (опционально)
      - image_base64: data-url или чистый base64  (или use_last=True + user_id)
      - width,height,steps,cfg,length (frames)
    Возвращает:
      - video_b64, path, seconds, last_image_path
    """
    m = _load_model()

    user_id  = str(inp.get("user_id", "u1"))
    use_last = bool(inp.get("use_last", False))
    prompt   = str(inp.get("prompt", ""))

    # входное изображение
    img: Optional[Image.Image] = None
    b64 = (inp.get("image_base64") or "").strip()
    if b64:
        img = _b64_to_image(b64)
        _save_last_image(user_id, img)
    elif use_last:
        img = _load_last_image(user_id)
        if img is None:
            raise RuntimeError("use_last=True but no previous image found; send image_base64 first.")
    else:
        raise RuntimeError("No image provided. Send image_base64 or set use_last=true after previous call.")

    width      = int(inp.get("width", 480))
    height     = int(inp.get("height", 832))
    steps      = int(inp.get("steps", 10))
    cfg_scale  = float(inp.get("cfg", 2.0))
    frame_num  = int(inp.get("length", 81))
    fps        = int(inp.get("fps", 24))

    t0 = time.time()
    # WanI2V ожидает PIL.Image, возвращает список/np массив кадров (T,H,W,3) uint8
    frames = m(
        img=img,
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=cfg_scale,
        frame_num=frame_num
    )
    secs = time.time() - t0

    # нормализуем в np.ndarray (T,H,W,3) uint8
    if isinstance(frames, (list, tuple)):
        arr = np.stack(frames, axis=0)
    else:
        arr = np.asarray(frames)
    if arr.ndim == 3:  # (T,H,W) -> (T,H,W,1) -> broadcast до 3 каналов при записи
        arr = arr[..., None]
    arr = arr.astype(np.uint8)

    out_path = RESULTS_DIR / f"wan22_{int(time.time())}.mp4"
    video_b64 = _arr_to_mp4_and_b64(arr, out_path, fps=fps)

    return {
        "video_b64": video_b64,
        "path": str(out_path),
        "seconds": round(secs, 3),
        "last_image_path": str(LAST_IMG_DIR / f"{user_id}.png"),
    }
