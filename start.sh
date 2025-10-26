#!/usr/bin/env bash
set -e

echo "[start] Patching Wan2.2 VAE backend if needed..."

python3 - <<'PY'
import os, re
from pathlib import Path

VAEP = Path("/app/Wan2.2/wan/modules/vae2_1.py")
src = VAEP.read_text(encoding="utf-8")

MARKER = "[to_empty|cpu-load|materialize|env]"
if MARKER not in src:
    append = r'''
# ---- Diffusers VAE backend (WAN if available, else BASE) [to_empty|cpu-load|materialize|env] ----
import os as _os, inspect as _inspect
import torch as _torch
from huggingface_hub import hf_hub_download
import json as _json

try:
    from diffusers import AutoencoderKLWan as _DiffVAE
    _WAN = True
except Exception:
    from diffusers import AutoencoderKL as _DiffVAE
    _WAN = False

def _filter_kwargs_for_ctor(cfg: dict, cls):
    import inspect
    allowed = set(inspect.signature(cls.__init__).parameters.keys()) - {"self","kwargs","**kwargs"}
    return {k: v for k, v in cfg.items() if k in allowed}

def _materialize_on_cpu(module):
    # materialize meta-params/buffers on CPU so .to(...) won't crash
    import torch.nn as _nn
    for name, p in list(module.named_parameters(recurse=True)):
        if getattr(p, "is_meta", False):
            mod = module
            parts = name.split(".")
            for part in parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], _nn.Parameter(_torch.empty_like(p, device="cpu"), requires_grad=False))
    for name, b in list(module.named_buffers(recurse=True)):
        if getattr(b, "is_meta", False):
            mod = module
            parts = name.split(".")
            for part in parts[:-1]:
                mod = getattr(mod, part)
            mod._buffers[parts[-1]] = _torch.empty_like(b, device="cpu")

def _vae_build_diffusers(_device):
    repo_id  = _os.environ.get("WAN_VAE_REPO", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    subf     = _os.environ.get("WAN_VAE_SUBFOLDER", "vae")
    fname    = _os.environ.get("WAN_VAE_FILENAME", "diffusion_pytorch_model.safetensors")
    token    = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")

    cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{subf}/config.json", token=token)
    w_path   = hf_hub_download(repo_id=repo_id, filename=f"{subf}/{fname}", token=token)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = _json.load(f)

    cfg = raw_cfg if _WAN else _filter_kwargs_for_ctor(raw_cfg, _DiffVAE)
    vae = _DiffVAE.from_config(cfg)  # create on CPU
    dtype = _torch.float16 if _device.type == "cuda" else _torch.float32

    used_to_empty = False
    if hasattr(vae, "to_empty"):
        try:
            vae = vae.to_empty(_device, dtype=dtype)
            used_to_empty = True
        except TypeError:
            try:
                vae = vae.to_empty(_device)
                used_to_empty = True
            except Exception:
                used_to_empty = False

    # load weights to CPU, filter by shape
    try:
        from safetensors.torch import load_file as _load_sf
        sd_full = _load_sf(w_path, device="cpu")
    except Exception:
        sd_full = _torch.load(w_path, map_location="cpu")

    if not used_to_empty:
        _materialize_on_cpu(vae)

    ms = vae.state_dict()
    sd = {k: v for k, v in sd_full.items() if k in ms and getattr(v, "shape", None) == getattr(ms[k], "shape", None)}
    missing = [k for k in ms.keys() if k not in sd]
    skipped = [k for k in sd_full.keys() if k not in sd]

    try:
        vae.load_state_dict(sd, strict=False, assign=True)
    except TypeError:
        vae.load_state_dict(sd, strict=False)

    if not used_to_empty:
        vae = vae.to(_device, dtype=dtype)
    else:
        try:
            vae = vae.to(dtype=dtype)
        except Exception:
            pass

    print(f"[VAE] diffusers load ({'WAN' if _WAN else 'BASE'}) strict=False: loaded={len(sd)} missing={len(missing)} skipped(mismatch)={len(skipped)}")
    vae.eval().requires_grad_(False)
    return vae

if _os.environ.get("USE_DIFFUSERS_VAE", "0") == "1":
    print("[VAE] Using diffusers VAE backend (WAN if present, else BASE) [to_empty|cpu-load|materialize|env]")
    _g = globals()
    _candidates = [k for k,v in _g.items() if _inspect.isclass(v) and "VAE" in k]
    for _name in _candidates:
        _cls = _g.get(_name)
        if not _cls or not hasattr(_cls, "__init__"):
            continue
        def _init(self, *a, **kw):
            import torch.nn as _nn
            try:
                _nn.Module.__init__(self)
            except Exception:
                pass
            _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self.model = _vae_build_diffusers(_device)
        _cls.__init__ = _init
        print("[VAE] patched class:", _name)
        break
'''
    src = src + "\n" + append

    # На всякий: если в коде был жёсткий repo_id — сделаем его ENV-driven
    src = re.sub(
        r'repo_id\s*=\s*"Wan-AI/Wan2\.2-TI2V-5B-Diffusers"',
        'repo_id = _os.environ.get("WAN_VAE_REPO", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")',
        src
    )

    VAEP.write_text(src, encoding="utf-8")
    print("[patch] inserted diffusers VAE backend with ENV control")
else:
    print("[patch] backend already present; skip")

PY

echo "[start] Runtime versions:"
python3 - <<'PY'
import os, torch
try:
    import diffusers; dv = diffusers.__version__
except Exception as e:
    dv = f"<not-importable: {e}>"
print("[runtime] torch", torch.__version__)
print("[runtime] diffusers", dv)
print("[runtime] USE_DIFFUSERS_VAE =", os.environ.get("USE_DIFFUSERS_VAE"))
print("[runtime] WAN_VAE_REPO     =", os.environ.get("WAN_VAE_REPO"))
print("[runtime] WAN_VAE_SUBFOLDER=", os.environ.get("WAN_VAE_SUBFOLDER"))
print("[runtime] WAN_VAE_FILENAME =", os.environ.get("WAN_VAE_FILENAME"))
PY

echo "[start] Launching handler..."
exec python3 -u /app/handler.py
