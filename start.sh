#!/usr/bin/env bash
set -euo pipefail

echo "[start] STARTING start.sh"
PYBIN="${PYBIN:-python3}"

###############################################################################
# 0) ПИН diffusers по желанию (чтобы не ребилдить образ)
#    Поставь DIFFUSERS_PIN=0.33 чтобы держать 0.33.x; иначе оставит текущую.
###############################################################################
if [[ "${DIFFUSERS_PIN:-}" =~ ^0\.33 ]]; then
  have="$(${PYBIN} - <<'PY'
import pkgutil, sys
try:
    import diffusers, json
    print(diffusers.__version__)
except Exception as e:
    print("NONE")
PY
)"
  if [[ "$have" != NONE && "$have" == 0.33.* ]]; then
    echo "[pin] diffusers is already $have"
  else
    echo "[pin] installing diffusers==0.33.0 ..."
    pip install -qU "diffusers==0.33.0"
  fi
fi

###############################################################################
# 1) Вывод версий и ENV — удобно для диагностики
###############################################################################
echo "[start] Runtime versions and env:"
${PYBIN} - <<'PY'
import torch, sys, os
try:
    import diffusers
    dv = diffusers.__version__
except Exception:
    dv = "N/A"
print("[runtime] torch", torch.__version__)
print("[runtime] diffusers", dv)
for k in ("USE_DIFFUSERS_VAE","WAN_VAE_REPO","WAN_VAE_SUBFOLDER",
          "WAN_VAE_FILENAME","WAN_LOW_NOISE_SUBFOLDER","HF_HOME"):
    if k in os.environ:
        print(f"[env] {k} = {os.environ[k]}")
PY

###############################################################################
# 2) ПАТЧ: diffusers VAE backend (WAN если есть, иначе BASE) + ENV override
#    а) поддержка WAN_VAE_REPO / WAN_VAE_SUBFOLDER / WAN_VAE_FILENAME
#    б) to_empty|cpu-load fallback + материализация meta-параметров на CPU
###############################################################################
echo "[patch] inserting diffusers VAE backend with ENV control and whitelist"
${PYBIN} - <<'PY'
from pathlib import Path
import os, re

tgt = Path("/app/Wan2.2/wan/modules/vae2_1.py")
s = tgt.read_text(encoding="utf-8")

marker = "Using diffusers VAE backend"
if marker not in s:
    s += """

# ===== DIFFUSERS VAE BACKEND (WAN if present, else BASE) [to_empty|cpu-load|materialize|env|whitelist] =====
import os as _os
import json as _json
import inspect as _inspect
import torch as _torch
from huggingface_hub import hf_hub_download

# 1) class selection
try:
    from diffusers import AutoencoderKLWan as _DiffVAE
    _WAN = True
except Exception:
    from diffusers import AutoencoderKL as _DiffVAE
    _WAN = False

def _filter_kwargs_for_ctor(cfg: dict, cls):
    allowed = set(_inspect.signature(cls.__init__).parameters.keys()) - {"self","kwargs","**kwargs"}
    return {k: v for k,v in cfg.items() if k in allowed}

def _materialize_on_cpu(module):
    \"\"\"Материализуем meta-параметры/буферы на CPU, чтобы .to(...) не падал.\"\"\"
    import torch.nn as _nn
    for name, p in list(module.named_parameters(recurse=True)):
        if getattr(p, "is_meta", False):
            mod = module
            parts = name.split(".")
            for part in parts[:-1]:
                mod = getattr(mod, part)
            last = parts[-1]
            new = _nn.Parameter(_torch.empty_like(p, device="cpu"), requires_grad=False)
            setattr(mod, last, new)
    for name, b in list(module.named_buffers(recurse=True)):
        if getattr(b, "is_meta", False):
            mod = module
            parts = name.split(".")
            for part in parts[:-1]:
                mod = getattr(mod, part)
            last = parts[-1]
            mod.register_buffer(last, _torch.empty_like(b, device="cpu"))

def _vae_build_diffusers(_device):
    # ENV-переопределения
    repo_id   = _os.environ.get("WAN_VAE_REPO",   "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    subfolder = _os.environ.get("WAN_VAE_SUBFOLDER", "vae")
    filename  = _os.environ.get("WAN_VAE_FILENAME",  "diffusion_pytorch_model.safetensors")

    token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/config.json",              token=token)
    w_path   = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/{filename}",               token=token)

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = _json.load(f)

    cfg = raw_cfg if _WAN else _filter_kwargs_for_ctor(raw_cfg, _DiffVAE)
    vae = _DiffVAE.from_config(cfg)  # CPU
    dtype = _torch.float16 if _device.type == "cuda" else _torch.float32

    # to_empty попытка
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

    # загружаем веса с фильтром по форме
    try:
        from safetensors.torch import load_file as _load_sf
        sd_full = _load_sf(w_path, device="cpu")
    except Exception:
        sd_full = _torch.load(w_path, map_location="cpu")

    ms = vae.state_dict()
    sd = {k: v for k, v in sd_full.items() if k in ms and getattr(v, "shape", None) == getattr(ms[k], "shape", None)}
    missing = [k for k in ms.keys() if k not in sd]
    skipped = [k for k in sd_full.keys() if k not in sd]

    try:
        vae.load_state_dict(sd, strict=False, assign=True)
    except TypeError:
        vae.load_state_dict(sd, strict=False)

    # если не было to_empty — переносим сейчас
    if not used_to_empty:
        _materialize_on_cpu(vae)
        vae = vae.to(_device, dtype=dtype)
    else:
        # если to_empty без dtype — доводим dtype
        try:
            vae = vae.to(dtype=dtype)
        except Exception:
            pass

    print(f"[VAE] diffusers load ({'WAN' if _WAN else 'BASE'}) strict=False: loaded={len(sd)} missing={len(missing)} skipped(mismatch)={len(skipped)}")
    vae.eval().requires_grad_(False)
    return vae

if _os.environ.get("USE_DIFFUSERS_VAE", "0") == "1":
    print("[VAE] Using diffusers VAE backend (WAN if present, else BASE) [to_empty|cpu-load|materialize|env|whitelist]")
    _g = globals()
    # Патчим классы VAE в этом модуле (Wan2_1_VAE / WanVAE / др.)
    for _name, _cls in list(_g.items()):
        if _inspect.isclass(_cls) and "VAE" in _name and hasattr(_cls, "__init__"):
            def _init(self, *a, **kw):
                _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                self.model = _vae_build_diffusers(_device)
            _cls.__init__ = _init
            print("[VAE] patched class:", _name)
    # ===== end DIFFUSERS backend =====
"""
    tgt.write_text(s, encoding="utf-8")
    print("[patch] vae2_1.py: diffusers backend injected")
else:
    print("[patch] vae2_1.py: backend already present")
PY

###############################################################################
# 3) ПАТЧ: Guard для .pth загрузки (skip при USE_DIFFUSERS_VAE=1, фильтр ключей)
###############################################################################
echo "[patch] guard native .pth load (indent-safe replacement)"
${PYBIN} - <<'PY'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[guard] file not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    if "import os" not in s:
        s = s.replace("import torch", "import torch\nimport os", 1)

    pat = re.compile(r"(?P<ind>\s*)missing,\s*mism\s*=\s*_load_filtered_state_dict\(\s*model\s*,\s*torch\.load\(\s*pretrained_path.*\)", re.M)
    if not pat.search(s):
        # вставим guard вокруг обычного load_state_dict(torch.load())
        s = re.sub(
            r"(?P<ind>\s*)model\.load_state_dict\(\s*torch\.load\((?P<pth>pretrained_path.*?)\)\s*(?:,\s*assign\s*=\s*True)?\s*\)",
            r"\g<ind>if os.environ.get('USE_DIFFUSERS_VAE','0')!='1' and os.path.isfile(pretrained_path):\n"
            r"\g<ind>    missing, mism = _load_filtered_state_dict(model, torch.load(\g<pth>, map_location=device))\n"
            r"\g<ind>else:\n\g<ind>    pass",
            s
        )
        p.write_text(s, encoding="utf-8")
        print("[guard] wrapped .pth load")
    else:
        print("[guard] already present")
PY

###############################################################################
# 4) ПАТЧ: low_noise_checkpoint — сделать необязательным
#    поддержка WAN_LOW_NOISE_SUBFOLDER и fallback без падения.
###############################################################################
echo "[patch] image2video.py: make low_noise_checkpoint optional"
${PYBIN} - <<'PY'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # A) где-то могли напрямую звать subfolder=config.low_noise_checkpoint
    s2 = re.sub(
        r"load_model\(\s*checkpoint_dir\s*,\s*subfolder\s*=\s*config\.low_noise_checkpoint\s*\)",
        "load_model(checkpoint_dir, subfolder=(os.environ.get('WAN_LOW_NOISE_SUBFOLDER') or getattr(config,'low_noise_checkpoint',None)))",
        s
    )
    if s2 != s:
        s = s2
        changed = True

    # B) нормализуем присваивание в self.low_noise = load_model(...)
    s2 = re.sub(
        r"(self\.low_noise\s*=\s*)load_model\(\s*checkpoint_dir\s*,\s*subfolder\s*=\s*(.+?)\)",
        r"_ln = (\2)\n\1load_model(checkpoint_dir, subfolder=_ln) if _ln else None",
        s
    )
    if s2 != s:
        s = s2
        changed = True

    if "import os" not in s:
        s = s.replace("import torch", "import torch\nimport os", 1)
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] low_noise optional: applied")
    else:
        print("[patch] low_noise optional: already patched")
PY

###############################################################################
# 5) Запускаем серверный обработчик
###############################################################################
echo "[start] Launching handler..."
exec ${PYBIN} -u /app/handler.py
