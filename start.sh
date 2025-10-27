#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[start][fatal] exit $?: at line $LINENO"; tail -n +1 -v /app/start.sh | nl | sed -n "$((LINENO-5)),$((LINENO+5))p"' ERR

echo "[start] STARTING start.sh"

# Basic runtime info
python3 - <<'PY1'
import sys, platform
print(f"[runtime] python {platform.python_version()}")
try:
    import torch as _t
    print(f"[runtime] torch {_t.__version__}")
except Exception as e:
    print("[runtime] torch: not found", e)
PY1

# --- Ensure diffusers >= 0.33.x (keep if already newer) ---
python3 - <<'PY2'
import subprocess, sys
def run(*args): subprocess.check_call([sys.executable, "-m", "pip", *args])
try:
    import diffusers, packaging.version as V
    v = diffusers.__version__
    print(f"[start] diffusers already {v}")
    if V.parse(v) < V.parse("0.33.0"):
        print("[start] upgrading diffusers to >=0.33.0 ...")
        run("install","-q","--upgrade","diffusers>=0.33.0")
except Exception:
    print("[start] installing diffusers >=0.33.0 ...")
    run("install","-q","--upgrade","diffusers>=0.33.0")
PY2

# --- Pin scientific stack to versions compatible with PyTorch/cu121 and diffusers ---
echo "[start][fix] pinning scientific stack (numpy/scipy/sklearn/numba) to compatible versions ..."
python3 - <<'PY3'
import subprocess, sys
def run(*args): subprocess.check_call([sys.executable, "-m", "pip", *args])
run("install","-q","--upgrade","--force-reinstall",
    "numpy<1.28",
    "scipy<1.12",
    "scikit-learn<1.4",
    "numba<0.61",
    "safetensors>=0.4.2")
import numpy, scipy, diffusers
print(f"[runtime-ok] numpy {numpy.__version__} / scipy {scipy.__version__} / diffusers {diffusers.__version__}")
PY3

# --- Show effective env used for VAE selection ---
echo "[env] USE_DIFFUSERS_VAE=${USE_DIFFUSERS_VAE:-0}"
echo "[env] WAN_VAE_REPO=${WAN_VAE_REPO:-<unset>}"
echo "[env] WAN_VAE_SUBFOLDER=${WAN_VAE_SUBFOLDER:-vae}"
echo "[env] WAN_VAE_FILENAME=${WAN_VAE_FILENAME:-diffusion_pytorch_model.safetensors}"

# --- Patch Wan2.2 VAE backend at runtime (idempotent) ---
echo "[patch] Patching Wan2.2 diffusers VAE backend (ENV-controlled) ..."
python3 - <<'PY4'
from pathlib import Path
import sys

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[patch][warn] file not found:", p)
    sys.exit(0)

s = p.read_text(encoding="utf-8")
marker = "diffusers VAE backend (WAN if present, else BASE) [to_empty|cpu-load|materialize|env|whitelist]"

if marker in s:
    print("[patch] backend already present — skip")
else:
    block = r"""
# --- diffusers VAE backend (WAN if present, else BASE) [to_empty|cpu-load|materialize|env|whitelist] ---
import os as _os, inspect as _inspect, json as _json
import torch as _torch
from huggingface_hub import hf_hub_download

# Try WAN class first and log result
try:
    from diffusers import AutoencoderKLWan as _DiffVAE
    print("[vae] WAN import = OK")
    _WAN = True
except Exception as _e:
    print(f"[vae] WAN import = FAIL -> BASE ({_e.__class__.__name__}: {_e})")
    from diffusers import AutoencoderKL as _DiffVAE
    _WAN = False

def _filter_kwargs_for_ctor(cfg: dict, cls):
    import inspect
    allowed = set(inspect.signature(cls.__init__).parameters.keys()) - {"self", "kwargs", "**kwargs"}
    return {k: v for k, v in cfg.items() if k in allowed}

def _materialize_on_cpu(module):
    """Instantiate meta-parameters/buffers on CPU so .to(...) does not crash."""
    import torch.nn as _nn
    for name, p in list(module.named_parameters(recurse=True)):
        if getattr(p, "is_meta", False):
            mod = module
            parts = name.split(".")
            for part in parts[:-1]:
                mod = getattr(mod, part)
            last = parts[-1]
            new = _nn.Parameter(_torch.empty_like(p, device=\"cpu\"), requires_grad=False)
            setattr(mod, last, new)
    for name, b in list(module.named_buffers(recurse=True)):
        if getattr(b, "is_meta", False):
            mod = module
            parts = name.split(\".\")
            for part in parts[:-1]:
                mod = getattr(mod, part)
            last = parts[-1]
            setattr(mod, last, _torch.empty_like(b, device=\"cpu\"))

def _vae_build_diffusers(_device):
    repo = _os.environ.get(\"WAN_VAE_REPO\",\"Wan-AI/Wan2.2-TI2V-5B-Diffusers\")
    sub  = _os.environ.get(\"WAN_VAE_SUBFOLDER\",\"vae\")
    fname= _os.environ.get(\"WAN_VAE_FILENAME\",\"diffusion_pytorch_model.safetensors\")
    token= _os.environ.get(\"HF_TOKEN\") or _os.environ.get(\"HUGGING_FACE_HUB_TOKEN\")
    cfg_path = hf_hub_download(repo_id=repo, filename=f\"{sub}/config.json\", token=token)
    w_path   = hf_hub_download(repo_id=repo, filename=f\"{sub}/{fname}\", token=token)
    with open(cfg_path,\"r\",encoding=\"utf-8\") as f:
        raw_cfg = _json.load(f)
    cfg = raw_cfg if _WAN else _filter_kwargs_for_ctor(raw_cfg, _DiffVAE)

    vae = _DiffVAE.from_config(cfg)              # start on CPU
    _materialize_on_cpu(vae)                     # ensure no meta tensors
    dtype = _torch.float16 if _device.type == \"cuda\" else _torch.float32

    used_to_empty = False
    if hasattr(vae, \"to_empty\"):
        try:
            vae = vae.to_empty(_device, dtype=dtype); used_to_empty = True
        except TypeError:
            try:
                vae = vae.to_empty(_device); used_to_empty = True
            except Exception:
                used_to_empty = False

    # load weights with shape filtering
    try:
        from safetensors.torch import load_file as _load_sf
        sd_full = _load_sf(w_path, device=\"cpu\")
    except Exception:
        sd_full = _torch.load(w_path, map_location=\"cpu\")

    ms = vae.state_dict()
    sd = {k:v for k,v in sd_full.items() if k in ms and getattr(v,\"shape\",None)==getattr(ms[k],\"shape\",None)}
    missing = [k for k in ms.keys() if k not in sd]
    skipped = [k for k in sd_full.keys() if k not in sd]

    try: vae.load_state_dict(sd, strict=False, assign=True)
    except TypeError: vae.load_state_dict(sd, strict=False)

    if not used_to_empty:
        vae = vae.to(_device, dtype=dtype)
    else:
        try: vae = vae.to(dtype=dtype)
        except Exception: pass

    print(f\"[VAE] diffusers load ({'WAN' if _WAN else 'BASE'}) strict=False: loaded={len(sd)} missing={len(missing)} skipped(mismatch)={len(skipped)}\")
    vae.eval().requires_grad_(False)
    return vae

if _os.environ.get(\"USE_DIFFUSERS_VAE\",\"0\") == \"1\":
    print(\"[VAE] Using diffusers VAE backend (WAN if present, else BASE) [to_empty|cpu-load|materialize|env|whitelist]\")
    _g = globals()
    # Patch the first class whose name contains 'VAE' (WanVAE/Wan2_1_VAE naming)
    for _name, _cls in list(_g.items()):
        if not hasattr(_cls, \"__name__\"): 
            continue
        if \"VAE\" in _cls.__name__ and hasattr(_cls, \"__init__\"):
            def _init(self, *a, **kw):
                device = _torch.device(\"cuda\" if _torch.cuda.is_available() else \"cpu\")
                self.model = _vae_build_diffusers(device)
            _cls.__init__ = _init
            print(\"[VAE] patched class:\", _cls.__name__)
            break
"""
    s = s + "\n" + block
    p.write_text(s, encoding="utf-8")
    print("[patch] backend appended")
PY4

echo "[start] Launching handler..."
exec python3 -u /app/handler.py
