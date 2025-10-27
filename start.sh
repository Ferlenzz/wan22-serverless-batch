#!/usr/bin/env bash
set -euo pipefail

echo "[start] STARTING start.sh"
PYBIN="${PYBIN:-python3}"

# -----------------------------------------------------------------------------
# PYTHONPATH: и /app (для sitecustomize.py), и /app/Wan2.2 (для wan.*)
# -----------------------------------------------------------------------------
export PYTHONPATH="/app:/app/Wan2.2:${PYTHONPATH:-}"

${PYBIN} - <<'PY'
from pathlib import Path
code = r'''
import builtins

def _patch_wani2v(mod):
    try:
        cls = getattr(mod, "WanI2V", None)
        if cls is None:
            return

        # 1) сделать класс вызываемым (__call__), делегируя на generate/infer/…
        if not hasattr(cls, "__call__"):
            def __call__(self, *args, **kwargs):
                for name in ("generate","infer","forward","run","predict","sample"):
                    fn = getattr(self, name, None)
                    if callable(fn):
                        return fn(*args, **kwargs)
                raise TypeError("WanI2V is not callable and no known generate-like method was found")
            cls.__call__ = __call__

        # 2) shim для generate: prompt/img -> позиционные, если args пусты
        gen = getattr(cls, "generate", None)
        if callable(gen) and not getattr(gen, "_shimmed", False):
            def _shim(self, *args, **kwargs):
                if not args:
                    prompt = kwargs.pop('prompt', kwargs.pop('input_prompt', ''))
                    img    = kwargs.pop('img', kwargs.pop('image', None))
                    return gen(self, prompt, img, **kwargs)
                return gen(self, *args, **kwargs)
            _shim._shimmed = True
            cls.generate = _shim

    except Exception:
        # fail-quietly
        pass

_orig_import = builtins.__import__
def _hook(name, globals=None, locals=None, fromlist=(), level=0):
    m = _orig_import(name, globals, locals, fromlist, level)
    try:
        # Патчим только ПОСЛЕ фактического импорта wan.image2video
        if name == "wan.image2video" or (name == "wan" and ("image2video" in (fromlist or ()) or fromlist == ("*",))):
            import sys as _sys
            mod = _sys.modules.get("wan.image2video")
            if mod is not None:
                _patch_wani2v(mod)
    except Exception:
        pass  # никогда ничего не печатаем
    return m

builtins.__import__ = _hook
'''
Path("/app/sitecustomize.py").write_text(code, encoding="utf-8")
print("[start] wrote ultra-quiet lazy sitecustomize.py (+generate shim)")
PY

# -----------------------------------------------------------------------------
# Диагностика (можно печатать — это уже не внутри sitecustomize.py)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYINFO'
import platform, os
try:
    import torch; tv = torch.__version__
except Exception: tv = "N/A"
try:
    import diffusers; dv = diffusers.__version__
except Exception: dv = "N/A"
print(f"[runtime] python {platform.python_version()}  torch {tv}  diffusers {dv}")
for k in ("USE_DIFFUSERS_VAE","WAN_VAE_REPO","WAN_VAE_SUBFOLDER","WAN_VAE_FILENAME",
          "WAN_LOW_NOISE_SUBFOLDER","WAN_HIGH_NOISE_SUBFOLDER","WAN_CKPT_DIR","HF_HOME","PYTHONPATH"):
    v = os.environ.get(k)
    if v is not None:
        print(f"[env] {k} = {v}")
PYINFO

# -----------------------------------------------------------------------------
# Гарантируем diffusers >= 0.35
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# vae2_1.py: инжект diffusers-бэкенда + жёсткий override _video_vae + guard .pth
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_VAE'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")

    # ensure import os
    if "import os" not in s:
        s = ("import os\n" + s) if "import torch" not in s else s.replace("import torch", "import torch\nimport os", 1)

    # inject diffusers backend (WAN if present)
    if "_vae_build_diffusers" not in s:
        s += """

# === Injected: diffusers VAE backend (WAN if present, else BASE), env-controlled ===
import json as _json, inspect as _inspect, torch as _torch
from huggingface_hub import hf_hub_download
def _filter_kwargs_for_ctor(cfg, cls):
    allowed = set(_inspect.signature(cls.__init__).parameters) - {"self"}
    return {k:v for k,v in cfg.items() if k in allowed}
def _vae_build_diffusers(_device):
    try:
        from diffusers import AutoencoderKLWan as _DiffVAE
        _WAN = True
    except Exception:
        from diffusers import AutoencoderKL as _DiffVAE
        _WAN = False
    repo=os.environ.get("WAN_VAE_REPO","Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    sub =os.environ.get("WAN_VAE_SUBFOLDER","vae")
    fn = os.environ.get("WAN_VAE_FILENAME","diffusion_pytorch_model.safetensors")
    tok=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cfg_path = hf_hub_download(repo_id=repo, filename=f"{sub}/config.json", token=tok)
    w_path   = hf_hub_download(repo_id=repo, filename=f"{sub}/{fn}", token=tok)
    with open(cfg_path,"r",encoding="utf-8") as f: raw=_json.load(f)
    vae = _DiffVAE.from_config(raw if _WAN else _filter_kwargs_for_ctor(raw,_DiffVAE))
    try:
        from safetensors.torch import load_file as _load_sf
        sd_full=_load_sf(w_path, device="cpu")
    except Exception:
        sd_full=_torch.load(w_path, map_location="cpu")
    ms=vae.state_dict()
    sd={k:v for k,v in sd_full.items() if k in ms and getattr(v,"shape",None)==getattr(ms[k],"shape",None)}
    try: vae.load_state_dict(sd, strict=False, assign=True)
    except TypeError: vae.load_state_dict(sd, strict=False)
    dtype=_torch.float16 if (_torch.cuda.is_available()) else _torch.float32
    try: vae = vae.to(dtype=dtype, device=_torch.device("cuda" if _torch.cuda.is_available() else "cpu"))
    except TypeError: vae = vae.to(_torch.device("cuda" if _torch.cuda.is_available() else "cpu"), dtype=dtype)
    print(f"[VAE] diffusers load ({'WAN' if _WAN else 'BASE'}) strict=False: loaded={len(sd)} missing={len(ms)-len(sd)} skipped(mismatch)={len(sd_full)-len(sd)}")
    vae.eval().requires_grad_(False)
    return vae
"""
        print("[patch] vae2_1.py: diffusers backend injected")

    # rename original _video_vae -> _video_vae_orig (one time)
    if "_video_vae_orig" not in s and "def _video_vae(" in s:
        s = s.replace("def _video_vae(", "def _video_vae_orig(", 1)
        print("[patch] vae2_1.py: renamed _video_vae -> _video_vae_orig")

    # hard override _video_vae (env-gated)
    if "def _video_vae(*_args, **_kwargs):" not in s:
        s += """

# === Injected: HARD OVERRIDE of _video_vae when USE_DIFFUSERS_VAE=1 ===
def _video_vae(*_args, **_kwargs):
    import os as __os, torch as __torch
    if __os.environ.get('USE_DIFFUSERS_VAE','0') == '1':
        _device = __torch.device('cuda' if __torch.cuda.is_available() else 'cpu')
        return _vae_build_diffusers(_device)
    if '_video_vae_orig' in globals():
        return globals()['_video_vae_orig'](*_args, **_kwargs)
    raise RuntimeError("Original _video_vae not available and USE_DIFFUSERS_VAE!=1")
"""
        print("[patch] vae2_1.py: hard override for _video_vae added")

    # guard any model.load_state_dict(torch.load(pretrained_path...)) when USE_DIFFUSERS_VAE=1
    s2 = re.sub(
        r"(?m)^(?P<ind>\s*)model\.load_state_dict\(\s*torch\.load\(\s*pretrained_path.*$",
        r"\g<ind>if os.environ.get('USE_DIFFUSERS_VAE','0')!='1' and os.path.isfile(pretrained_path):\n"
        r"\g<ind>    model.load_state_dict(torch.load(pretrained_path, map_location=device))\n"
        r"\g<ind>else:\n\g<ind>    print('[VAE] skip native .pth load:', pretrained_path)",
        s
    )
    if s2 != s:
        s = s2
        print("[patch] vae2_1.py: guarded native .pth load")

    p.write_text(s, encoding="utf-8")
PYPATCH_VAE

# -----------------------------------------------------------------------------
# image2video.py: делаем low/high_noise_checkpoint опциональными (ENV-aware)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_NOISE'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/image2video.py")
if p.exists():
    s = p.read_text(encoding="utf-8")
    changed = False

    if "import os" not in s:
        s = ("import os\n" + s) if "import torch" not in s else s.replace("import torch", "import torch\nimport os", 1)
        changed = True

    def patch_noise(s, name, env, prefix):
        ch = False
        s2 = re.sub(rf"subfolder\s*=\s*config\.{name}",
                    rf"subfolder=(os.environ.get('{env}') or getattr(config,'{name}',None))", s)
        if s2 != s: s, ch = s2, True
        s2 = re.sub(
            rf"(?P<ind>\s*)self\.{prefix}_noise\s*=\s*load_model\(\s*checkpoint_dir\s*,\s*subfolder\s*=\s*(.+?)\)",
            rf"\g<ind>_n = (\2)\n\g<ind>self.{prefix}_noise = load_model(checkpoint_dir, subfolder=_n) if _n else None",
            s
        )
        if s2 != s: s, ch = s2, True
        return s, ch

    s, ch = patch_noise(s, "low_noise_checkpoint",  "WAN_LOW_NOISE_SUBFOLDER",  "low");  changed |= ch
    s, ch = patch_noise(s, "high_noise_checkpoint", "WAN_HIGH_NOISE_SUBFOLDER", "high"); changed |= ch

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: low/high optional (env-aware)")
    else:
        print("[patch] image2video.py: already ok")
else:
    print("[patch][warn] not found:", p)
PYPATCH_NOISE

# -----------------------------------------------------------------------------
# vae2_1.py: совместимость с diffusers AutoencoderKLOutput в encode()
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_ENCODE'
from pathlib import Path, re

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # Найдём место, где внутри encode() делается ... self.model.encode(...).float().squeeze(0)
    # и заменим на логику распаковки AutoencoderKLOutput.
    pat = re.compile(
        r"""(self\.model\.encode\(\s*u\.unsqueeze\(0\)\s*,\s*self\.scale\s*\)\.float\(\)\.squeeze\(0\))""",
        re.VERBOSE
    )
    if pat.search(s):
        s = pat.sub(
            "(_out := self.model.encode(u.unsqueeze(0), self.scale), "
            "_tmp := (getattr(_out, 'latent_dist', None).mode() if getattr(_out, 'latent_dist', None) is not None "
            "else (getattr(_out, 'latents', None) if getattr(_out, 'latents', None) is not None else _out)), "
            "_tmp.float().squeeze(0))[-1]",
            s
        )
        changed = True
    else:
        # запасной вариант — если без .squeeze(0) или формат чуть другой
        pat2 = re.compile(
            r"""(self\.model\.encode\(\s*u\.unsqueeze\(0\)\s*,\s*self\.scale\s*\)\.float\(\))""",
            re.VERBOSE
        )
        if pat2.search(s):
            s = pat2.sub(
                "(_out := self.model.encode(u.unsqueeze(0), self.scale), "
                "_tmp := (getattr(_out, 'latent_dist', None).mode() if getattr(_out, 'latent_dist', None) is not None "
                "else (getattr(_out, 'latents', None) if getattr(_out, 'latents', None) is not None else _out)), "
                "_tmp.float())[-1]",
                s
            )
            changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] vae2_1.py: encode() now unwraps AutoencoderKLOutput")
    else:
        print("[patch] vae2_1.py: encode() patch not applied (pattern not found) — maybe already patched?")
PYPATCH_ENCODE

# -----------------------------------------------------------------------------
# image2video.py: поддержка boundary как dict (enabled/lower/upper) и как числа — INDENT-AWARE
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_BOUNDARY'
from pathlib import Path
import re

p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # Ищем строку вроде: boundary = self.boundary * self.num_train_timesteps
    # (допускаем пробелы/их отсутствие вокруг '*')
    pat = re.compile(
        r'^(?P<ind>\s*)boundary\s*=\s*self\.boundary\s*\*\s*self\.num_train_timesteps\s*$',
        re.M
    )

    def repl(m):
        ind  = m.group('ind')               # исходный базовый отступ
        ind1 = ind + "    "                 # +1 уровень вложенности
        return (
            f"{ind}# normalize boundary (dict or scalar)\n"
            f"{ind}_b = self.boundary\n"
            f"{ind}_n = self.num_train_timesteps\n"
            f"{ind}if isinstance(_b, dict) or hasattr(_b, 'get'):\n"
            f"{ind1}if not (_b.get('enabled', False)):\n"
            f"{ind1}    boundary = None\n"
            f"{ind1}else:\n"
            f"{ind1}    _lo = int(max(0, min(1, float(_b.get('lower', 0.0)))) * _n)\n"
            f"{ind1}    _up = int(max(0, min(1, float(_b.get('upper', 1.0)))) * _n)\n"
            f"{ind1}    boundary = (_lo, _up)\n"
            f"{ind}else:\n"
            f"{ind1}try:\n"
            f"{ind1}    boundary = int(float(_b) * _n)\n"
            f"{ind1}except Exception:\n"
            f"{ind1}    boundary = None"
        )

    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2
        changed = True
    else:
        # запасной паттерн без пробелов вокруг '*'
        pat2 = re.compile(
            r'^(?P<ind>\s*)boundary\s*=\s*self\.boundary\*\s*self\.num_train_timesteps\s*$',
            re.M
        )
        s3 = pat2.sub(repl, s)
        if s3 != s:
            s = s3
            changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: boundary handling normalized (indent-aware)")
    else:
        print("[patch] image2video.py: boundary patch not applied (pattern not found) — maybe already patched?")
PYPATCH_BOUNDARY

# -----------------------------------------------------------------------------
# image2video.py: безопасное сравнение с boundary (может быть None | int | tuple)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_BOUNDARY_CMP'
from pathlib import Path
import re

p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # Ищем условие вида: if t.item() >= boundary:
    pat = re.compile(r'^(?P<ind>\s*)if\s+t\.item\(\)\s*>=\s*boundary\s*:\s*$', re.M)

    def repl(m):
        ind  = m.group('ind')
        ind1 = ind + "    "
        return (
            f"{ind}# safe compare with boundary that may be None | int | (lo, up)\n"
            f"{ind}_b = boundary\n"
            f"{ind}if _b is None:\n"
            f"{ind1}_cond = False\n"
            f"{ind}elif isinstance(_b, tuple):\n"
            f"{ind1}try:\n"
            f"{ind1}    _lo, _up = _b\n"
            f"{ind1}except Exception:\n"
            f"{ind1}    _lo, _up = 0, int(_b[1]) if len(_b)>1 else 0\n"
            f"{ind1}_cond = (t.item() >= _up)\n"
            f"{ind}else:\n"
            f"{ind1}try:\n"
            f"{ind1}    _cond = (t.item() >= int(_b))\n"
            f"{ind1}except Exception:\n"
            f"{ind1}    _cond = False\n"
            f"{ind}if _cond:"
        )

    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: safe boundary comparison applied")
    else:
        print("[patch] image2video.py: comparison patch not applied (pattern not found) — maybe already patched?")
PYPATCH_BOUNDARY_CMP

# -----------------------------------------------------------------------------
# image2video.py: безопасный выбор sample_guide_scale при boundary=None|int|tuple
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_GUIDE'
from pathlib import Path, re

p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # Шаблоны тернарника: guide_scale[1] if t.item() >= boundary else guide_scale[0]
    pat = re.compile(
        r'^(?P<ind>\s*)sample_guide_scale\s*=\s*guide_scale\[\s*1\s*\]\s*'
        r'if\s*t\.item\(\)\s*>=\s*boundary\s*else\s*guide_scale\[\s*0\s*\]\s*$',
        re.M
    )
    def repl(m):
        ind  = m.group('ind')
        ind1 = ind + "    "
        return (
            f"{ind}# safe guide scale pick with boundary (None|int|(lo,up))\n"
            f"{ind}_b = boundary\n"
            f"{ind}if _b is None:\n"
            f"{ind1}_cond = False\n"
            f"{ind}elif isinstance(_b, tuple):\n"
            f"{ind1}try:\n"
            f"{ind1}    _lo, _up = _b\n"
            f"{ind1}except Exception:\n"
            f"{ind1}    _lo, _up = 0, int(_b[1]) if len(_b)>1 else 0\n"
            f"{ind1}_cond = (t.item() >= _up)\n"
            f"{ind}else:\n"
            f"{ind1}try:\n"
            f"{ind1}    _cond = (t.item() >= int(_b))\n"
            f"{ind1}except Exception:\n"
            f"{ind1}    _cond = False\n"
            f"{ind}sample_guide_scale = guide_scale[1] if _cond else guide_scale[0]"
        )

    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2
        changed = True
    else:
        # запасной — если без пробелов/переносов
        pat2 = re.compile(
            r'^(?P<ind>\s*)sample_guide_scale\s*=\s*guide_scale\[1\]\s*if\s*t\.item\(\)\s*>=\s*boundary\s*else\s*guide_scale\[0\]\s*$',
            re.M
        )
        s3 = pat2.sub(repl, s)
        if s3 != s:
            s = s3
            changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: safe sample_guide_scale selection applied")
    else:
        print("[patch] image2video.py: guide-scale ternary not found (maybe already patched?)")
PYPATCH_GUIDE

# -----------------------------------------------------------------------------
# Запуск хэндлера
# -----------------------------------------------------------------------------
echo "[start] Launching handler..."
exec ${PYBIN} -u /app/handler.py
