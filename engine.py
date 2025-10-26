
import os
import io
import base64
from typing import Dict, Any, Optional
from pathlib import Path

from loguru import logger
from PIL import Image
import numpy as np
import imageio

# ------------------ WAN 2.2 ------------------
# Конфиг как ОБЪЕКТ (EasyDict), его НЕ нужно "вызвать".
from wan.configs.wan_ti2v_5B import ti2v_5B as cfg  # EasyDict | иногда старая ревизия = callable
from wan.image2video import WanI2V

# ------------------ Paths & ENV ------------------
WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B")
RP_VOLUME = os.environ.get("RP_VOLUME", "/runpod-volume")
LAST_DIR = Path(RP_VOLUME) / "state" / "last_images"
LAST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Model singleton ------------------
_MODEL: Optional[WanI2V] = None

def _ensure_cfg_boundary():
    """Добавить boundary, если его нет в конфиге (требуется WanI2V.__init__)."""
    global cfg
    # если вдруг в твоей копии это функция — вызовем и заменим на объект
    if callable(cfg):
        logger.warning("ti2v_5B is callable in this revision; calling ti2v_5B() to obtain config object.")
        cfg_obj = cfg()
    else:
        cfg_obj = cfg
    if not hasattr(cfg_obj, "boundary"):
        # дефолт, который работает стабильно
        cfg_obj.boundary = 500
    return cfg_obj

def _model() -> WanI2V:
    global _MODEL
    if _MODEL is None:
        cfg_obj = _ensure_cfg_boundary()
        logger.info(f"Loading WanI2V with checkpoints from: {WAN_CKPT_DIR}")
        _MODEL = WanI2V(cfg_obj, WAN_CKPT_DIR, device_id=0)
        logger.info("WanI2V ready.")
    return _MODEL

# ------------------ Utils ------------------
def _b64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return im

def _image_to_b64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _save_last_image(user_id: str, im: Image.Image):
    path = LAST_DIR / f"{user_id}.png"
    try:
        im.save(path, format="PNG")
    except Exception as e:
        logger.warning(f"Failed to save last image for {user_id}: {e}")

def _load_last_image(user_id: str) -> Optional[Image.Image]:
    path = LAST_DIR / f"{user_id}.png"
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load last image for {user_id}: {e}")
    return None

def _frames_to_mp4_b64(frames: np.ndarray, fps: int = 24) -> str:
    """frames: (T, H, W, 3) uint8 -> data:video/mp4;base64,..."""
    buf = io.BytesIO()
    # imageio-ffmpeg backend (imageio==2.34+ тянет ffmpeg из системы)
    with imageio.get_writer(buf, format="FFMPEG", mode="I", fps=fps, codec="libx264", macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()

# ------------------ Public API ------------------
def generate_one(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Главная точка входа для handler.py.

    Ожидаемые поля в inp:
      - action: str ('gen'|'warmup'|'health' ... ) — не обязателен, тут игнорим
      - prompt: str
      - user_id: str
      - use_last: bool
      - image_base64: str (data:image/...)
      - width, height: int (необязательно, используется для max_area)
      - steps: int
      - cfg: float
      - length: int (frame_num)
    Возвращает:
      dict(status='COMPLETED', output={'video': 'data:video/mp4;base64,...', 'mime': 'video/mp4'})
    """
    m = _model()

    prompt = inp.get("prompt", "")
    user_id = inp.get("user_id", "u1")
    use_last = bool(inp.get("use_last", False))
    b64 = (inp.get("image_base64") or "").strip()

    # подготовим картинку: либо присланную, либо last
    img: Optional[Image.Image] = None
    if b64:
        img = _b64_to_image(b64)
        # сохраним, чтобы в следующий раз работал use_last
        _save_last_image(user_id, img)
    elif use_last:
        img = _load_last_image(user_id)
        if img is None:
            raise RuntimeError("use_last=True, но прошлой картинки нет. Пришлите image_base64.")

    if img is None:
        raise RuntimeError("Не передано изображение (image_base64 пуст) и use_last=False.")

    steps = int(inp.get("steps", 40))
    guide = float(inp.get("cfg", 5.0))
    frame_num = int(inp.get("length", 81))
    # max_area ограничивает суммарные пиксели. Если передали width/height — используем их произведение
    width = int(inp.get("width", 0) or 0)
    height = int(inp.get("height", 0) or 0)
    max_area = int(width * height) if (width > 0 and height > 0) else 921600  # 1280*720 ~ 921600

    # WanI2V.generate сигнатура (актуальная на твоей сборке):
    # (input_prompt, img, max_area=..., frame_num=..., shift=..., sample_solver='unipc',
    #  sampling_steps=..., guide_scale=..., n_prompt='', seed=-1, offload_model=True)
    logger.info(f"Generate: len={frame_num}, steps={steps}, cfg={guide}, max_area={max_area}, prompt='{prompt[:80]}'")
    video = m.generate(
        input_prompt=prompt,
        img=img,
        max_area=max_area,
        frame_num=frame_num,
        sampling_steps=steps,
        guide_scale=guide,
        offload_model=True
    )

    # video -> np.uint8[T,H,W,3]
    if isinstance(video, list):
        arr = np.stack([np.asarray(v) for v in video], axis=0)
    else:
        arr = np.asarray(video)
        if arr.ndim == 4 and arr.shape[-1] in (1,3):
            pass
        elif arr.ndim == 3:
            # T,H,W -> T,H,W,1
            arr = arr[..., None]
    arr = arr.astype(np.uint8)

    b64mp4 = _frames_to_mp4_b64(arr, fps=24)
    return {
        "status": "COMPLETED",
        "output": {
            "video": b64mp4,
            "mime": "video/mp4"
        }
    }
