#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native-style engine for RunPod serverless worker.
See docstring inside for details.
"""
import base64, binascii, io, os, shutil, subprocess, time, uuid
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image

RP_VOLUME   = os.environ.get("RP_VOLUME", "/runpod-volume")
STATE_DIR   = os.path.join(RP_VOLUME, "state")
LAST_DIR    = os.path.join(STATE_DIR, "last_images")
RESULTS_DIR = os.path.join(RP_VOLUME, "results")
TMP_DIR     = os.path.join(RP_VOLUME, "tmp")

WAN_CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B")
WAN_VAE_PATH = os.environ.get("WAN_VAE_PATH", "/runpod-volume/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")

os.makedirs(LAST_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

@dataclass
class GenParams:
    prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    length: int
    user_id: str
    image_b64: Optional[str] = None
    use_last: bool = False
    return_video_b64: bool = True

def _now_ms() -> int:
    return int(time.time() * 1000)

def _b64_to_pil(img_b64: str) -> Image.Image:
    if img_b64.startswith("data:image"):
        _, b64 = img_b64.split(",", 1)
    else:
        b64 = img_b64
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _encode_video_to_b64(mp4_path: str) -> str:
    with open(mp4_path, "rb") as f:
        raw = f.read()
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("utf-8")

def _save_last_image(user_id: str, img: Image.Image) -> str:
    path = os.path.join(LAST_DIR, f"{user_id}.png")
    img.save(path, format="PNG")
    return path

def _load_last_image(user_id: str) -> Optional[Image.Image]:
    path = os.path.join(LAST_DIR, f"{user_id}.png")
    if os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None

def run_wan_native(model_dir: str, vae_path: str, prompt: str,
                   ref_image_path: str, width: int, height: int,
                   steps: int, cfg: float, length: int, out_mp4: str) -> None:
    """
    TODO: ВСТАВЬ реальный вызов WAN 2.2 (Python API или CLI) здесь.
    Примеры и подсказки в README_WORKFLOW.md в репозитории.
    Сейчас — осознанный raise, чтобы не молча "успешно" ничего не делать.
    """
    raise FileNotFoundError(
        "Вставьте ваш реальный вызов WAN 2.2 в run_wan_native(...). "
        f"Модель: {model_dir}, VAE: {vae_path}"
    )

def generate_one(params: GenParams, job_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if job_id is None:
        job_id = str(uuid.uuid4())

    start_ms = _now_ms()
    src_img: Optional[Image.Image] = None

    if params.image_b64:
        try:
            src_img = _b64_to_pil(params.image_b64)
        except binascii.Error:
            src_img = None

    if src_img is None and params.use_last:
        src_img = _load_last_image(params.user_id)

    if src_img is None:
        src_img = Image.new("RGB", (params.width, params.height), (255, 255, 255))

    _save_last_image(params.user_id, src_img)

    work_dir = os.path.join(TMP_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)
    ref_png_path = os.path.join(work_dir, "ref.png")
    src_img.save(ref_png_path, format="PNG")

    out_mp4_path = os.path.join(RESULTS_DIR, f"{job_id}.mp4")

    run_wan_native(
        model_dir=WAN_CKPT_DIR,
        vae_path=WAN_VAE_PATH,
        prompt=params.prompt,
        ref_image_path=ref_png_path,
        width=params.width,
        height=params.height,
        steps=params.steps,
        cfg=params.cfg,
        length=params.length,
        out_mp4=out_mp4_path,
    )

    video_b64 = _encode_video_to_b64(out_mp4_path) if params.return_video_b64 else None
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    exec_ms = _now_ms() - start_ms
    return out_mp4_path, video_b64

def generate(payload: dict) -> dict:
    t0 = _now_ms()
    action = str(payload.get("action", "gen"))

    if action == "warmup":
        return {
            "ok": True,
            "model_dir": WAN_CKPT_DIR,
            "vae_path": WAN_VAE_PATH,
            "model_exists": os.path.exists(WAN_CKPT_DIR),
            "vae_exists": os.path.exists(WAN_VAE_PATH),
            "executionTime": _now_ms() - t0,
        }

    params = GenParams(
        prompt=str(payload.get("prompt", "a cinematic shot")),
        width=int(payload.get("width", 480)),
        height=int(payload.get("height", 832)),
        steps=int(payload.get("steps", 10)),
        cfg=float(payload.get("cfg", 2.0)),
        length=int(payload.get("length", 81)),
        user_id=str(payload.get("user_id", "u1")),
        image_b64=payload.get("image_base64"),
        use_last=bool(payload.get("use_last", False)),
        return_video_b64=bool(payload.get("return_video_b64", True)),
    )

    job_id = str(uuid.uuid4())
    try:
        mp4_path, b64 = generate_one(params, job_id=job_id)
        out = {"ok": True, "job_id": job_id, "mp4_path": mp4_path, "executionTime": _now_ms() - t0}
        if b64:
            out["video"] = b64
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "executionTime": _now_ms() - t0}
