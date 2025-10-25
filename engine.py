import os, json, time, base64, subprocess, requests
from pathlib import Path
from typing import Optional, Tuple

RP_VOLUME      = os.environ.get("RP_VOLUME", "/runpod-volume")
WAN_DIR        = os.environ.get("WAN_CKPT_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B")
WAN_VAE        = os.environ.get("WAN_VAE_PATH", f"{WAN_DIR}/Wan2.2_VAE.pth")
WORKFLOW_JSON  = os.environ.get("WORKFLOW_JSON", "/runpod-volume/workflows/new_Wan22_api.json")

COMFY_DIR      = "/app/ComfyUI"
COMFY_API      = "http://127.0.0.1:8188"

LAST_DIR       = Path(RP_VOLUME) / "state" / "last_images"
RESULTS_DIR    = Path(RP_VOLUME) / "results"
OUTPUT_DIR     = Path(COMFY_DIR) / "output"

LAST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_comfy_started = False

def _start_comfy_once():
    global _comfy_started
    if _comfy_started:
        return
    try:
        requests.get(COMFY_API, timeout=0.3)
        _comfy_started = True
        return
    except Exception:
        pass

    subprocess.Popen(
        ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--no-gpu-check"],
        cwd=COMFY_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    for _ in range(60):
        try:
            requests.get(COMFY_API, timeout=0.5)
            _comfy_started = True
            break
        except Exception:
            time.sleep(0.5)
    if not _comfy_started:
        raise RuntimeError("ComfyUI API not reachable")

def _b64_of_image_path(p: Path) -> str:
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def _choose_input_image(user_id: str, image_base64: Optional[str], use_last: bool) -> Optional[str]:
    if image_base64:
        return image_base64
    if use_last:
        p = LAST_DIR / f"{user_id}.png"
        if p.exists():
            return _b64_of_image_path(p)
    return None

def _set_input(workflow: dict, **slots):
    def visit(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    visit(v)
                else:
                    if k in slots and slots[k] is not None:
                        d[k] = slots[k]
        elif isinstance(d, list):
            for it in d: visit(it)
    visit(workflow)
    return workflow

def generate_one(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    length: int,
    user_id: str = "u1",
    use_last: bool = True,
    image_base64: Optional[str] = None,
    return_base64: bool = False
) -> dict:
    """Генерация одного видео через ComfyUI/Wan2.2.
    Возвращает { ok, mp4_path, video_base64?, exec_ms }.
    """
    t0 = time.time()
    _start_comfy_once()

    img_b64 = _choose_input_image(user_id, image_base64, use_last)
    if not os.path.exists(WORKFLOW_JSON):
        raise FileNotFoundError(f"Workflow JSON not found: {WORKFLOW_JSON}")

    with open(WORKFLOW_JSON, "r", encoding="utf-8") as f:
        wf = json.load(f)

    slots = dict(
        prompt=prompt,
        width=width, height=height, steps=steps, cfg=cfg, length=length,
        image_base64=img_b64,
        model_dir=WAN_DIR,
        vae_path=WAN_VAE,
    )
    wf = _set_input(wf, **slots)

    r = requests.post(f"{COMFY_API}/prompt", json=wf, timeout=60)
    r.raise_for_status()
    # ждём появления нового mp4 в output/
    seen = set(p.name for p in OUTPUT_DIR.glob("*.mp4"))
    result_mp4 = None
    for _ in range(1200):
        for p in OUTPUT_DIR.glob("*.mp4"):
            if p.name not in seen:
                result_mp4 = p
                break
        if result_mp4:
            break
        time.sleep(0.25)
    if not result_mp4:
        raise RuntimeError("ComfyUI: mp4 not produced")

    job_id = str(int(time.time() * 1000))
    dst = RESULTS_DIR / f"{job_id}.mp4"
    result_mp4.replace(dst)

    out = {"ok": True, "mp4_path": str(dst), "exec_ms": int((time.time()-t0)*1000)}

    if return_base64:
        with open(dst, "rb") as f:
            out["video_base64"] = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
    return out
