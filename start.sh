#!/usr/bin/env bash
set -euo pipefail

echo "[start] STARTING start.sh"
PYBIN="${PYBIN:-python3}"

# --- Runtime info
${PYBIN} - <<'PYINFO'
import platform
print(f"[runtime] python {platform.python_version()}")
try:
    import torch
    print(f"[runtime] torch {torch.__version__}")
except Exception as e:
    print("[runtime] torch not importable:", e)
PYINFO

# --- Ensure diffusers >= 0.35 (do NOT downgrade)
${PYBIN} - <<'PYDIF'
import sys, subprocess
from packaging.version import Version
def run(*args): subprocess.check_call([sys.executable,"-m","pip",*args])
try:
    import diffusers
    v = Version(diffusers.__version__)
    print(f"[start] diffusers detected: {diffusers.__version__}")
    if v < Version("0.35.0"):
        print("[start] upgrading diffusers to >=0.35.0 ...")
        run("install","-q","--upgrade","diffusers>=0.35.0")
except Exception:
    print("[start] installing diffusers>=0.35.0 ...")
    run("install","-q","--upgrade","diffusers>=0.35.0")
PYDIF

# --- Echo important ENV
${PYBIN} - <<'PYENV'
import os
for k in ("USE_DIFFUSERS_VAE","WAN_VAE_REPO","WAN_VAE_SUBFOLDER",
          "WAN_VAE_FILENAME","WAN_LOW_NOISE_SUBFOLDER","WAN_CKPT_DIR"):
    print(f"[env] {k} =", os.environ.get(k))
PYENV

# --- Patch vae2_1.py: inject diffusers backend and HARD OVERRIDE _video_vae (skip .pth when USE_DIFFUSERS_VAE=1)
${PYBIN} - <<'PYPATCH1'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
s = p.read_text(encoding="utf-8")

# Ensure imports
if "import os" not in s:
    s = s.replace("import torch", "import torch\nimport os", 1)

# Inject diffusers backend builder if not present
if "_vae_build_diffusers" not in s:
    s += r"""

# === Injected: diffusers VAE backend (WAN if present else BASE), env-controlled ===
import json as _json
import inspect as _inspect
import torch as _torch
from huggingface_hub import hf_hub_download

def _filter_kwargs_for_ctor(cfg: dict, cls):
    import inspect as __insp
    allowed = set(__insp.signature(cls.__init__).parameters.keys()) - {"self","kwargs","**kwargs"}
    return {k:v for k,v in cfg.items() if k in allowed}

def _vae_build_diffusers(_device):
    # select class
    try:
        from diffusers import AutoencoderKLWan as _DiffVAE
        _WAN = True
    except Exception:
        from diffusers import AutoencoderKL as _DiffVAE
        _WAN = False

    repo   = os.environ.get("WAN_VAE_REPO",   "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    sub    = os.environ.get("WAN_VAE_SUBFOLDER","vae")
    fname  = os.environ.get("WAN_VAE_FILENAME","diffusion_pytorch_model.safetensors")
    token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    cfg_path = hf_hub_download(repo_id=repo, filename=f"{sub}/config.json", token=token)
    w_path   = hf_hub_download(repo_id=repo, filename=f"{sub}/{fname}",     token=token)
    with open(cfg_path,"r",encoding="utf-8") as f:
        raw_cfg = _json.load(f)

    # build CPU model
    if _WAN:
        vae = _DiffVAE.from_config(raw_cfg)
    else:
        vae = _DiffVAE.from_config(_filter_kwargs_for_ctor(raw_cfg, _DiffVAE))

    # load weights with shape filter
    try:
        from safetensors.torch import load_file as _load_sf
        sd_full = _load_sf(w_path, device="cpu")
    except Exception:
        import torch as __t
        sd_full = __t.load(w_path, map_location="cpu")

    ms = vae.state_dict()
    sd = {k:v for k,v in sd_full.items() if k in ms and getattr(v,"shape",None)==getattr(ms[k],"shape",None)}
    missing = [k for k in ms.keys() if k not in sd]
    skipped = [k for k in sd_full.keys() if k not in sd]
    try:
        vae.load_state_dict(sd, strict=False, assign=True)
    except TypeError:
        vae.load_state_dict(sd, strict=False)

    import torch as __t
    dtype = __t.float16 if (_device.type=="cuda") else __t.float32
    try:
        vae = vae.to(dtype=dtype, device=_device)
    except TypeError:
        vae = vae.to(_device, dtype=dtype)

    print(f"[VAE] diffusers load ({'WAN' if _WAN else 'BASE'}) strict=False: loaded={len(sd)} missing={len(missing)} skipped(mismatch)={len(skipped)}")
    vae.eval().requires_grad_(False)
    return vae
"""
    print("[patch] injected _vae_build_diffusers")

# Rename original _video_vae once (to keep fallback), then add hard override
if "_video_vae_orig" not in s and "def _video_vae(" in s:
    s = s.replace("def _video_vae(", "def _video_vae_orig(", 1)
    print("[patch] renamed original _video_vae -> _video_vae_orig")

if "def _video_vae(*_args, **_kwargs):" not in s:
    s += r"""

# === Injected: HARD OVERRIDE of _video_vae when USE_DIFFUSERS_VAE=1 ===
def _video_vae(*_args, **_kwargs):
    import torch as __torch
    if os.environ.get('USE_DIFFUSERS_VAE','0') == '1':
        _device = __torch.device('cuda' if __torch.cuda.is_available() else 'cpu')
        return _vae_build_diffusers(_device)
    # fallback to original behavior if env not set
    if '_video_vae_orig' in globals():
        return globals()['_video_vae_orig'](*_args, **_kwargs)
    raise RuntimeError("Original _video_vae not available and USE_DIFFUSERS_VAE!=1")
"""
    print("[patch] added hard override for _video_vae")

p.write_text(s, encoding="utf-8")
print("[patch] vae2_1.py saved")
PYPATCH1

# --- Make low_noise_checkpoint optional (ENV WAN_LOW_NOISE_SUBFOLDER or None)
${PYBIN} - <<'PYPATCH2'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch] image2video.py not found")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    s2 = re.sub(
        r"load_model\(\s*checkpoint_dir\s*,\s*subfolder\s*=\s*config\.low_noise_checkpoint\s*\)",
        "load_model(checkpoint_dir, subfolder=(os.environ.get('WAN_LOW_NOISE_SUBFOLDER') or getattr(config,'low_noise_checkpoint', None)))",
        s
    )
    if s2 != s:
        s = s2; changed = True

    s2 = re.sub(
        r"(self\.low_noise\s*=\s*)load_model\(\s*checkpoint_dir\s*,\s*subfolder\s*=\s*(.+?)\)",
        r"_ln = (\2)\n\1load_model(checkpoint_dir, subfolder=_ln) if _ln else None",
        s
    )
    if s2 != s:
        s = s2; changed = True

    if "import os" not in s:
        s = s.replace("import torch", "import torch\nimport os", 1)
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py updated (low_noise optional)")
    else:
        print("[patch] image2video.py already ok")
PYPATCH2

echo "[start] Launching handler..."
exec ${PYBIN} -u /app/handler.py
