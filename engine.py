import os
import io
import time
import base64
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from inspect import signature, Parameter

# --- ensure interpreter sees /app and /app/Wan2.2 ---
for p in ("/app", "/app/Wan2.2"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from loguru import logger
from PIL import Image
import numpy as np
import imageio

# ------------------ Wan 2.2 ------------------
from wan.configs.wan_ti2v_5B import ti2v_5B as cfg
from wan.image2video import WanI2V

# --- EasyDict compatibility ---
try:
    from easydict import EasyDict as EDict
except Exception:
    class EDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

# ----- paths -----
RP_VOLUME   = Path(os.environ.get("RP_VOLUME", "/runpod-volume"))
CKPT_DIR    = Path(os.environ.get("WAN_CKPT_DIR", RP_VOLUME / "models" / "Wan2.2-TI2V-5B"))
RESULTS_DIR = RP_VOLUME / "results"
LAST_IMG_DIR= RP_VOLUME / "last_images"
for d in (RESULTS_DIR, LAST_IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)

_MODEL: Optional[WanI2V] = None

# ================= helpers =================
def _ensure_cfg():
    """Take ti2v_5B config and normalize missing fields."""
    c = cfg() if callable(cfg) else cfg

    # guarantee boundary block presence
    if not hasattr(c, "boundary") or c.boundary is None:
        c.boundary = EDict({
            "enabled": False,
            "lower": 0.0,
            "upper": 1.0,
        })
    else:
        # normalize key name enable -> enabled
        if isinstance(c.boundary, dict):
            if "enable" in c.boundary and "enabled" not in c.boundary:
                c.boundary["enabled"] = bool(c.boundary.get("enable"))
        else:
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
    """arr: (T,H,W,3) uint8 -> save to out_path and return data:video/mp4;base64,..."""
    imageio.mimsave(out_path, arr, fps=fps, macro_block_size=None)
    with open(out_path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

# ---- make WanI2V safely callable ----
def _get_wan_call(obj):
    call = getattr(obj, "__call__", None)
    if callable(call):
        return call
    for name in ("generate","infer","forward","run","predict","sample"):
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    raise TypeError("WanI2V is not callable and no known generate method found")

def _filter_and_remap_kwargs(fn, kw: Dict[str, Any]) -> Dict[str, Any]:
    """Фильтруем keyword-параметры под сигнатуру fn и мапим синонимы."""
    sig = signature(fn)
    params = sig.parameters

    def has(name: str) -> bool:
        p = params.get(name)
        return p is not None and p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)

    out: Dict[str, Any] = {}

    # image/img → keyword, если метод это поддерживает как keyword
    v_img = kw.get("img", kw.get("image"))
    if v_img is not None:
        if has("image"):
            out["image"] = v_img
        elif has("img"):
            out["img"] = v_img

    # width/height
    for k in ("width", "height"):
        if k in kw and has(k):
            out[k] = kw[k]

    # steps
    if "steps" in kw:
        if has("steps"):
            out["steps"] = kw["steps"]
        elif has("num_inference_steps"):
            out["num_inference_steps"] = kw["steps"]

    # guidance scale (cfg)
    v_cfg = kw.get("guidance_scale", kw.get("cfg", kw.get("cfg_scale")))
    if v_cfg is not None:
        if has("guidance_scale"):
            out["guidance_scale"] = v_cfg
        elif has("scale"):
            out["scale"] = v_cfg

    # frames / length
    v_frames = kw.get("frame_num", kw.get("length", kw.get("frames", kw.get("num_frames"))))
    if v_frames is not None:
        for name in ("frame_num","num_frames","frames","length"):
            if has(name):
                out[name] = v_frames
                break

    # prompt → только если метод явно принимает keyword с таким именем
    if "prompt" in kw:
        if has("prompt"):
            out["prompt"] = kw["prompt"]
        elif has("input_prompt"):
            out["input_prompt"] = kw["prompt"]

    # fps (если поддерживается keyword'ом)
    if "fps" in kw and has("fps"):
        out["fps"] = kw["fps"]

    return out

def _invoke_wan(obj, **kw):
    """Вызывает метод модели, поддерживая позиционно-обязательные аргументы."""
    fn = _get_wan_call(obj)
    sig = signature(fn)
    params = list(sig.parameters.values())

    # Подготовим кандидатов
    img_val = kw.get("img", kw.get("image"))
    prompt_val = kw.get("prompt", kw.get("input_prompt"))

    # 1) Соберём позиционные аргументы (POSITIONAL_ONLY)
    args = []
    for p in params:
        if p.kind is Parameter.POSITIONAL_ONLY:
            if p.name in ("input_prompt", "prompt", "text"):
                args.append(prompt_val)
            elif p.name in ("img", "image"):
                args.append(img_val)
            else:
                # другой позиционный — попытаемся взять из kw по имени
                args.append(kw.get(p.name))
        elif p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY, Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            break

    # Проверка наличия обязательных позиционных без default
    need_pos = [p for p in params if p.kind is Parameter.POSITIONAL_ONLY and p.default is Parameter.empty]
    if any(a is None for a in args[:len(need_pos)]):
        raise TypeError("Required positional-only params are missing; make sure to pass image_base64 and (optionally) prompt.")

    # 2) Именованные (keyword) — фильтруем и мапим
    kwargs = _filter_and_remap_kwargs(fn, kw)

    return fn(*args, **kwargs)

class _WanI2VProxy:
    def __init__(self, obj): self._obj = obj
    def __getattr__(self, n): return getattr(self._obj, n)
    def __call__(self, *a, **k): return _invoke_wan(self._obj, **k)

def _wrap_wani2v(obj): return _WanI2VProxy(obj)

# ================ public API =================
def warmup() -> Dict[str, Any]:
    _ = _load_model()
    return {"ok": True, "message": "model initialized"}

def generate_one(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a short video from an input image."""
    m = _wrap_wani2v(_load_model())

    user_id  = str(inp.get("user_id", "u1"))
    use_last = bool(inp.get("use_last", False))
    prompt   = str(inp.get("prompt", ""))

    # input image
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
    frames = _invoke_wan(
        m,
        img=img,
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=cfg_scale,
        frame_num=frame_num,
        fps=fps,
    )
    secs = time.time() - t0

    # normalize to np.ndarray (T,H,W,3) uint8
    if isinstance(frames, (list, tuple)):
        arr = np.stack(frames, axis=0)
    else:
        arr = np.asarray(frames)
    if arr.ndim == 3:  # (T,H,W) -> (T,H,W,1)
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
