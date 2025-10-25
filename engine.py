import base64, io, os, time
from typing import Dict, Any, List

from PIL import Image
import numpy as np
import imageio.v2 as imageio
import torch

from loguru import logger

import sys
sys.path.append('/app/Wan2.2')
from wan.configs.wan_ti2v_5B import ti2v_5B as cfg_fn
from wan.image2video import WanI2V

CKPT_DIR = os.environ.get('WAN_CKPT_DIR', '/runpod-volume/models/Wan2.2-TI2V-5B')
RP_VOLUME = os.environ.get('RP_VOLUME', '/runpod-volume')

_MODEL = None

def _init_model() -> WanI2V:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    cfg = cfg_fn()

    if not hasattr(cfg, 'boundary'):
        cfg.boundary = 500
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
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
        return [f.numpy() for f in x]
    raise TypeError(f"Unsupported video type: {type(v)}")

def generate_one(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Main generation routine used by handler.
    slots keys:
      - prompt: str
      - image_base64: str
      - length: int (frames), default 81
      - steps: int, default 40
      - cfg: float, default 5.0
      - width,height (optional): only used to compute max_area (defaults 1280x720)
    returns dict: { ok: bool, video_b64: str, seconds: float, path: str }
    """
    m = _init_model()

    prompt = slots.get('prompt') or ''
    image_b64 = slots.get('image_base64')
    if not image_b64:
        return {'ok': False, 'error': 'image_base64 is required'}

    img = _b64_to_pil(image_b64)

    width = int(slots.get('width', 1280))
    height = int(slots.get('height', 720))
    max_area = width * height

    frame_num = int(slots.get('length', 81))
    steps = int(slots.get('steps', 40))
    guide_scale = float(slots.get('cfg', 5.0))

    t0 = time.time()
    video = m.generate(
        input_prompt=prompt,
        img=img,
        frame_num=frame_num,
        sampling_steps=steps,
        guide_scale=guide_scale,
        max_area=max_area,
        offload_model=True,
    )
    frames = _to_numpy_frames(video)
    mp4_bytes = _frames_to_mp4(frames, fps=24)
    import base64 as _b64
    b64 = _b64.b64encode(mp4_bytes).decode('utf-8')
    secs = time.time() - t0

    os.makedirs(f"{RP_VOLUME}/results", exist_ok=True)
    out_path = os.path.join(RP_VOLUME, 'results', f'wan22_{int(time.time())}.mp4')
    with open(out_path, 'wb') as f:
        f.write(mp4_bytes)

    return {
        'ok': True,
        'video_b64': 'data:video/mp4;base64,' + b64,
        'seconds': round(secs, 3),
        'path': out_path,
    }
