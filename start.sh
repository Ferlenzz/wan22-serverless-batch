#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[start][fatal] exit $?: at line $LINENO"; tail -n +1 -v /app/start.sh | nl | sed -n "$((LINENO-5)),$((LINENO+5))p"' ERR

echo "[start] STARTING start.sh"

# --- runtime info
python3 - <<'PYINFO'
import platform
print(f"[runtime] python {platform.python_version()}")
try:
    import torch
    print(f"[runtime] torch {torch.__version__}")
except Exception as e:
    print("[runtime] torch not importable:", e)
PYINFO

# --- ensure diffusers >= 0.33 (WAN support present in newer versions)
python3 - <<'PYDIF'
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
PYDIF

# --- pin scientific stack (avoid NumPy 2.x ABI breaks etc.)
echo "[start][fix] pinning scientific stack (numpy/scipy/sklearn/numba) ..."
python3 - <<'PYPIN'
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
PYPIN

# --- env echo
echo "[env] USE_DIFFUSERS_VAE=${USE_DIFFUSERS_VAE:-0}"
echo "[env] WAN_VAE_REPO=${WAN_VAE_REPO:-<unset>}"
echo "[env] WAN_VAE_SUBFOLDER=${WAN_VAE_SUBFOLDER:-vae}"
echo "[env] WAN_VAE_FILENAME=${WAN_VAE_FILENAME:-diffusion_pytorch_model.safetensors}"

# --- disable local pth if using diffusers VAE
if [ "${USE_DIFFUSERS_VAE:-0}" = "1" ]; then
  VDIR="${WAN_CKPT_DIR:-/runpod-volume/models/Wan2.2-TI2V-5B}"
  if [ -f "$VDIR/Wan2.2_VAE.pth" ]; then
    mv "$VDIR/Wan2.2_VAE.pth" "$VDIR/Wan2.2_VAE.pth.disabled" || true
    echo "[start] disabled local VAE pth at $VDIR"
  fi
fi

# --- backend-injection patch (separate script to avoid heredoc quoting issues)
cat >/tmp/patch_vae.py <<'PY'
from pathlib import Path
import os, importlib, inspect
from huggingface_hub import hf_hub_download
import torch as _torch
import json as _json

def _wan_import():
    try:
        from diffusers import AutoencoderKLWan as _DiffVAE
        print("[vae] WAN import = OK")
        return _DiffVAE, True
    except Exception as e:
        from diffusers import AutoencoderKL as _DiffVAE
        print(f"[vae] WAN import = FAIL -> BASE ({e.__class__.__name__}: {e})")
        return _DiffVAE, False

def _filter_kwargs(cfg, cls):
    allowed = set(inspect.signature(cls.__init__).parameters.keys()) - {'self','kwargs','**kwargs'}
    return {k:v for k,v in cfg.items() if k in allowed}

def _materialize_on_cpu(module):
    import torch.nn as _nn
    for name, p in list(module.named_parameters(recurse=True)):
        if getattr(p, 'is_meta', False):
            mod = module
            parts = name.split('.')
            for part in parts[:-1]: mod = getattr(mod, part)
            setattr(mod, parts[-1], _nn.Parameter(_torch.empty_like(p, device='cpu'), requires_grad=False))
    for name, b in list(module.named_buffers(recurse=True)):
        if getattr(b, 'is_meta', False):
            mod = module
            parts = name.split('.')
            for part in parts[:-1]: mod = getattr(mod, part)
            mod._buffers[parts[-1]] = _torch.empty_like(b, device='cpu')

def _vae_build_diffusers(device):
    _DiffVAE, _WAN = _wan_import()
    repo = os.environ.get('WAN_VAE_REPO','Wan-AI/Wan2.2-TI2V-5B-Diffusers')
    sub  = os.environ.get('WAN_VAE_SUBFOLDER','vae')
    fname= os.environ.get('WAN_VAE_FILENAME','diffusion_pytorch_model.safetensors')
    token= os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

    cfg_path = hf_hub_download(repo_id=repo, filename=f"{sub}/config.json", token=token)
    w_path   = hf_hub_download(repo_id=repo, filename=f"{sub}/{fname}", token=token)
    with open(cfg_path,'r',encoding='utf-8') as f:
        raw_cfg = _json.load(f)
    cfg = raw_cfg if _WAN else _filter_kwargs(raw_cfg, _DiffVAE)

    vae = _DiffVAE.from_config(cfg)      # CPU init
    _materialize_on_cpu(vae)
    dtype = _torch.float16 if device.type=='cuda' else _torch.float32

    used_to_empty = False
    if hasattr(vae,'to_empty'):
        try:
            vae = vae.to_empty(device, dtype=dtype); used_to_empty=True
        except TypeError:
            try:
                vae = vae.to_empty(device); used_to_empty=True
            except Exception:
                used_to_empty=False

    try:
        from safetensors.torch import load_file as _load_sf
        sd_full = _load_sf(w_path, device='cpu')
    except Exception:
        sd_full = _torch.load(w_path, map_location='cpu')

    ms = vae.state_dict()
    sd = {k:v for k,v in sd_full.items() if k in ms and getattr(v,'shape',None)==getattr(ms[k],'shape',None)}
    missing = [k for k in ms.keys() if k not in sd]
    skipped = [k for k in sd_full.keys() if k not in sd]

    try: vae.load_state_dict(sd, strict=False, assign=True)
    except TypeError: vae.load_state_dict(sd, strict=False)

    if not used_to_empty:
        vae = vae.to(device, dtype=dtype)
    else:
        try: vae = vae.to(dtype=dtype)
        except Exception: pass

    print(f"[VAE] diffusers load ({'WAN' if _WAN else 'BASE'}) strict=False: loaded={len(sd)} missing={len(missing)} skipped(mismatch)={len(skipped)}")
    vae.eval().requires_grad_(False)
    return vae

def apply_patch():
    if os.environ.get('USE_DIFFUSERS_VAE','0')!='1':
        print("[patch] USE_DIFFUSERS_VAE=0 — skip")
        return
    m = importlib.import_module('wan.modules.vae2_1')
    for name, cls in list(vars(m).items()):
        if isinstance(cls, type) and 'VAE' in getattr(cls,'__name__','') and hasattr(cls,'__init__'):
            def _init(self, *a, **kw):
                device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
                self.model = _vae_build_diffusers(device)
            setattr(cls, '__init__', _init)
            print("[VAE] patched class:", cls.__name__)
            break

apply_patch()
PY

python3 /tmp/patch_vae.py

# --- guard native .pth load (indent-safe replacement; no f-strings)
python3 - <<'PY'
from pathlib import Path
import re, os
p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[guard][warn] file not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    if "import os" not in s:
        s = s.replace("import torch", "import torch\nimport os", 1)

    pat = re.compile(r'^(?P<ind>\s*)missing,\s*mism\s*=\s*_load_filtered_state_dict\(\s*model\s*,\s*torch\.load\(\s*pretrained_path.*$', re.M)
    def repl(m):
        ind  = m.group('ind')
        ind1 = ind + "    "
        lines = [
            f"{ind}if os.environ.get('USE_DIFFUSERS_VAE','0')=='1' or not os.path.isfile(pretrained_path):",
            f"{ind1}print('[VAE] skip .pth load (USE_DIFFUSERS_VAE or missing file):', pretrained_path)",
            f"{ind}else:",
            f"{ind1}missing, mism = _load_filtered_state_dict(model, torch.load(pretrained_path, map_location=device))",
        ]
        return "\n".join(lines)
    s2, n = pat.subn(repl, s)
    if n > 0:
        p.write_text(s2, encoding="utf-8")
        print(f"[guard] wrapped .pth load in {p} (occurrences: {n})")
    else:
        print("[guard] target .pth load line not found — maybe already patched")
PY

# --- ensure 'import os' and guard native .pth load ---
python3 - <<'PY'
from pathlib import Path
import re

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[guard][warn] file not found:", p)
else:
    s = p.read_text(encoding="utf-8")

    # 1) гарантируем import os сразу после первого import torch
    if "import os" not in s.split("\n", 40):  # смотрим только верх файла
        s = s.replace("import torch", "import torch\nimport os", 1)
        print("[guard] injected 'import os' after 'import torch'")

    # 2) жёсткий гард вокруг model.load_state_dict(torch.load(...pretrained_path...))
    #    если USE_DIFFUSERS_VAE=1 ИЛИ файла нет — пропускаем torch.load
    pat = re.compile(
        r'^(?P<ind>\s*)missing,\s*mism\s*=\s*_load_filtered_state_dict\(\s*model,\s*torch\.load\(\s*pretrained_path.*$',
        re.M
    )
    def repl(m):
        ind  = m.group('ind')
        code = (
            f"{ind}if os.environ.get('USE_DIFFUSERS_VAE','0') == '1' or not os.path.isfile(pretrained_path):\n"
            f"{ind}    print('[VAE] skip native .pth load (env=USE_DIFFUSERS_VAE or missing file):', pretrained_path)\n"
            f"{ind}else:\n"
            f"{ind}    missing, mism = _load_filtered_state_dict(model, torch.load(pretrained_path, map_location=device))"
        )
        return code

    s2 = pat.sub(repl, s)
    if s2 != s:
        p.write_text(s2, encoding="utf-8")
        print("[guard] wrapped native .pth load in", p)
    else:
        # запасной паттерн, если строка без нашего _load_filtered_state_dict(...)
        pat2 = re.compile(
            r'^(?P<ind>\s*)model\.load_state_dict\(\s*torch\.load\(\s*pretrained_path.*$',
            re.M
        )
        def repl2(m):
            ind  = m.group('ind')
            return (
                f"{ind}if os.environ.get('USE_DIFFUSERS_VAE','0') == '1' or not os.path.isfile(pretrained_path):\n"
                f"{ind}    print('[VAE] skip native .pth load (env=USE_DIFFUSERS_VAE or missing file):', pretrained_path)\n"
                f"{ind}else:\n"
                f"{ind}    " + m.group(0).strip()
            )
        s3 = pat2.sub(repl2, s)
        if s3 != s:
            p.write_text(s3, encoding="utf-8")
            print("[guard] wrapped model.load_state_dict(torch.load(...)) in", p)
        else:
            print("[guard] target .pth load not found — возможно уже пропатчено")
PY

echo "[start] Launching handler..."
exec python3 -u /app/handler.py
