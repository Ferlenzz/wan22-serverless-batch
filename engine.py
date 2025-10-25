import base64, io, os, time
from typing import Dict, Any, List

from PIL import Image
import numpy as np
import imageio.v2 as imageio
import torch
from loguru import logger

# Wan2.2 imports
import sys
sys.path.append('/app/Wan2.2')
from wan.configs.wan_ti2v_5B import ti2v_5B as cfg_fn
from wan.image2video import WanI2V

CKPT_DIR = os.environ.get('WAN_CKPT_DIR', '/runpod-volume/models/Wan2.2-TI2V-5B')
RP_VOLUME = os.environ.get('RP_VOLUME', '/runpod-volume')

RESULTS_DIR = os.path.join(RP_VOLUME, 'results')
LAST_DIR    = os.path.join(RP_VOLUME, 'state', 'last_images')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LAST_DIR, exist_ok=True)

_MODEL = None

def _init_model() -> WanI2V:
    """Lazy init WAN 2.2 I2V (с VAE 2.2)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    cfg = cfg_fn()
    if not hasattr(cfg, 'boundary'):
        cfg.boundary = 500  # безопасное дефолт-значение
    _MODEL = WanI2V(cfg, CKPT_DIR, device_id=0)
    logger.info("WanI2V initialized with checkpoints at {}", CKPT_DIR)
    return _MODEL

def _b64_to_pil(b64: str) -> Image.Image:
    if b64.startswith('data:'):
        b64 = b64.split(',', 1)[1]
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert('RGB')

def _frames_to_mp4(frames: List[np.ndarray], fps: int = 24) -> bytes:
    buf = io.BytesIO()
    imageio.mimwrite(buf, frames, format='ffmpeg', fps=fps, codec='libx264', quality=8)
    return buf.getvalue()

def _to_numpy_frames(v) -> List[np.ndarray]:
    if isinstance(v, list):
        return [np.array(img) for img in v]
    if isinstance(v, torch.Tensor):
        x = v.detach().cpu()
        if x.max() <= 1.0:
            x = x * 255.0
        x = x.clamp(0, 255).byte()
        # [T,C,H,W] -> [T,H,W,C]
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
        return [f.numpy() for f in x]
    raise TypeError(f"Unsupported video type: {type(v)}")

def _load_last_image(user_id: str) -> Image.Image | None:
    p = os.path.join(LAST_DIR, f"{user_id}.png")
    if os.path.exists(p):
        try:
            return Image.open(p).convert('RGB')
        except Exception as e:
            logger.warning("Failed to load last image {}: {}", p, e)
    return None

def _save_last_image(arr: np.ndarray, user_id: str) -> str:
    """Сохраняем последний кадр как PNG для следующего запроса."""
    p = os.path.join(LAST_DIR, f"{user_id}.png")
    Image.fromarray(arr).save(p, format='PNG')
    return p

def generate_one(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Входные поля:
      - prompt: str
      - image_base64: str (если пусто и use_last=True — возьмём из /state/last_images/<user_id>.png)
      - use_last: bool
      - user_id: str (имя файла last-image), по умолчанию 'u1'
      - length: int (frames), default 81
      - steps: int, default 40
      - cfg: float, default 5.0
      - width,height (для оценки max_area; по умолчанию 1280x720)
    Возврат: { ok, video_b64, seconds, path, last_image_path }
    """
    m = _init_model()

    prompt   = slots.get('prompt') or ''
    user_id  = str(slots.get('user_id') or 'u1')
    use_last = bool(slots.get('use_last', False))

    image_b64 = slots.get('image_base64')
    img: Image.Image | None = None

    if image_b64:
        img = _b64_to_pil(image_b64)
    elif use_last:
        img = _load_last_image(user_id)
        if img is None:
            return {'ok': False, 'error': f'No last image for user_id={user_id}. Provide image_base64 or set use_last=false.'}
    else:
        return {'ok': False, 'error': 'image_base64 is required unless use_last=true.'}

    width  = int(slots.get('width', 1280))
    height = int(slots.get('height', 720))
    max_area = width * height

    frame_num   = int(slots.get('length', 81))     # дешевле 81; можно 49-65 для ускорения
    steps       = int(slots.get('steps', 40))      # 32–40 — хороший баланс
    guide_scale = float(slots.get('cfg', 5.0))     # 4.0–6.0

    t0 = time.time()
    video = m.generate(
        input_prompt=prompt,
        img=img,
        frame_num=frame_num,
        sampling_steps=steps,
        guide_scale=guide_scale,
        max_area=max_area,
        offload_model=True,   # дешевле по памяти
    )

    frames = _to_numpy_frames(video)
    # сохраняем last_image (последний кадр)
    last_img_path = _save_last_image(frames[-1], user_id)

    mp4_bytes = _frames_to_mp4(frames, fps=24)
    b64 = base64.b64encode(mp4_bytes).decode('utf-8')
    secs = time.time() - t0

    out_path = os.path.join(RESULTS_DIR, f'wan22_{int(time.time())}.mp4')
    with open(out_path, 'wb') as f:
        f.write(mp4_bytes)

    return {
        'ok': True,
        'video_b64': 'data:video/mp4;base64,' + b64,
        'seconds': round(secs, 3),
        'path': out_path,
        'last_image_path': last_img_path,
    }
