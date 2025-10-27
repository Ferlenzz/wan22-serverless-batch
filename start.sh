#!/usr/bin/env bash
set -euo pipefail

echo "[start] STARTING start.sh"
PYBIN="${PYBIN:-python3}"

# -----------------------------------------------------------------------------
# PYTHONPATH: и /app (для sitecustomize.py), и /app/Wan2.2 (для wan.*)
# -----------------------------------------------------------------------------
export PYTHONPATH="/app:/app/Wan2.2:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# sitecustomize.py (тихий):
#  - делает WanI2V вызываемым (__call__)
#  - shim для generate: нормализует self.boundary и маппит prompt/img по именам
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY'
from pathlib import Path
code = r'''
# не печатать из этого файла
import builtins, inspect

def _normalize_boundary(self):
    try:
        b = getattr(self, 'boundary', None)
        n = int(getattr(self, 'num_train_timesteps', 1000))
        if isinstance(b, dict) or hasattr(b, 'get'):
            if not b.get('enabled', False):
                return None
            lo = int(max(0, min(1, float(b.get('lower', 0.0)))) * n)
            up = int(max(0, min(1, float(b.get('upper', 1.0)))) * n)
            return (lo, up)
        if b is None:
            return None
        return int(float(b) * n) if isinstance(b, (int, float, str)) else None
    except Exception:
        return None

def _patch_wani2v(mod):
    try:
        cls = getattr(mod, "WanI2V", None)
        if cls is None:
            return

        # 1) сделать класс вызываемым (__call__), делегируя на generate/infer/…
        if not hasattr(cls, "__call__"):
            def __call__(self, *a, **k):
                for name in ("generate","infer","forward","run","predict","sample"):
                    fn = getattr(self, name, None)
                    if callable(fn):
                        return fn(*a, **k)
                raise TypeError("WanI2V is not callable and no known generate-like method was found")
            cls.__call__ = __call__

        # 2) generate-shim: нормализуем boundary и маппим именованные аргументы
        gen = getattr(cls, "generate", None)
        if callable(gen) and not getattr(gen, "_shimmed_kwmap_boundary", False):
            sig = inspect.signature(gen)
            params = sig.parameters
            def _shim(self, *args, **kwargs):
                nb = _normalize_boundary(self)
                try: object.__setattr__(self, 'boundary', nb)
                except Exception: setattr(self, 'boundary', nb)

                if not args:
                    prompt = kwargs.pop('prompt', kwargs.pop('input_prompt', ''))
                    img    = kwargs.pop('img', kwargs.pop('image', None))
                    call_kwargs = dict(kwargs)

                    if 'prompt' in params: call_kwargs['prompt'] = prompt
                    elif 'input_prompt' in params: call_kwargs['input_prompt'] = prompt

                    if img is not None:
                        if 'img' in params: call_kwargs['img'] = img
                        elif 'image' in params: call_kwargs['image'] = img
                        elif 'x' in params: call_kwargs['x'] = [img]

                    return gen(self, **call_kwargs)
                return gen(self, *args, **kwargs)
            _shim._shimmed_kwmap_boundary = True
            cls.generate = _shim

    except Exception:
        pass

_orig_import = builtins.__import__
def _hook(name, globals=None, locals=None, fromlist=(), level=0):
    m = _orig_import(name, globals, locals, fromlist, level)
    try:
        if name == "wan.image2video" or (name == "wan" and ("image2video" in (fromlist or ()) or fromlist == ("*",))):
            import sys as _sys
            mod = _sys.modules.get("wan.image2video")
            if mod is not None:
                _patch_wani2v(mod)
    except Exception:
        pass
    return m

builtins.__import__ = _hook
'''
Path("/app/sitecustomize.py").write_text(code, encoding="utf-8")
print("[start] wrote sitecustomize.py (generate shim + boundary normalization)")
PY

# -----------------------------------------------------------------------------
# Диагностика
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
# Гарантируем diffusers >= 0.35 (не даунгрейдим)
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
# vae2_1.py: diffusers VAE backend + hard override _video_vae + guard .pth
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_VAE'
from pathlib import Path
import re, os

p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")

    if "import os" not in s:
        s = ("import os\n" + s) if "import torch" not in s else s.replace("import torch", "import torch\nimport os", 1)

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

    if "_video_vae_orig" not in s and "def _video_vae(" in s:
        s = s.replace("def _video_vae(", "def _video_vae_orig(", 1)
        print("[patch] vae2_1.py: renamed _video_vae -> _video_vae_orig")

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
# image2video.py: low/high_noise_checkpoint опциональны (ENV-aware)
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
# image2video.py: ГЛОБАЛЬНЫЙ безопасный патч сравнения с boundary + локальная boundary
# -----------------------------------------------------------------------------
${PYBIN} - <<'PYPATCH_BOUNDARY_GLOBAL'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) Вставим safe helper один раз (после первых импортов)
    helper = r"""
# --- safe boundary compare helper (None | int | tuple(lo, up)) ---
def _cmp_ge_boundary(t_val, boundary):
    if boundary is None:
        return False
    if isinstance(boundary, tuple):
        try:
            _, up = boundary
        except Exception:
            up = int(boundary[1]) if hasattr(boundary,'__len__') and len(boundary) > 1 else 0
        try:
            return t_val >= int(up)
        except Exception:
            return False
    try:
        return t_val >= int(boundary)
    except Exception:
        return False
"""
    if "_cmp_ge_boundary(" not in s:
        s = re.sub(r"^(import[^\n]+\n(?:from[^\n]+\n)*)", r"\1"+helper+"\n", s, count=1, flags=re.M)
        changed = True

    # 2) Тернарники "guide_scale[1] if t.item() >= boundary else guide_scale[0]"
    s2 = re.sub(
        r"guide_scale\[\s*1\s*\]\s*if\s*t\.item\(\)\s*>=\s*boundary\s*else\s*guide_scale\[\s*0\s*\]",
        r"guide_scale[1] if _cmp_ge_boundary(t.item(), boundary) else guide_scale[0]",
        s
    )
    if s2 != s: s = s2; changed = True

    # 3) Любые "if t.item() >= boundary:"
    s2 = re.sub(
        r"if\s+t\.item\(\)\s*>=\s*boundary\s*:",
        r"if _cmp_ge_boundary(t.item(), boundary):",
        s
    )
    if s2 != s: s = s2; changed = True

    # 4) В начале generate(): локальная нормализация boundary
    s2 = re.sub(
        r"(?ms)^(\s*def\s+generate\s*\(.*?\):\s*\n)(\s*)(\S)",
        lambda m: f"{m.group(1)}{m.group(2)}boundary = getattr(self, 'boundary', None)\n{m.group(2)}{m.group(3)}",
        s
    )
    if s2 != s: s = s2; changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: global safe boundary compare + local boundary init")
    else:
        print("[patch] image2video.py: boundary globals already applied")
PYPATCH_BOUNDARY_GLOBAL

# -----------------------------------------------------------------------------
# Запуск хэндлера
# -----------------------------------------------------------------------------
echo "[start] Launching handler..."
exec ${PYBIN} -u /app/handler.py
