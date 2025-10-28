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
#  - shim для generate: маппит prompt/img в именованные параметры
#  - WAN_FORCE_BOUNDARY_OFF=1 мягко отключает boundary в рантайме
#  - WAN_FORCE_FRAMES=N: жёстко выставляет кадры в объекте/конфиге + пробрасывает kw
# -----------------------------------------------------------------------------

${PYBIN} - <<'PY_SITE'
from pathlib import Path
code = r'''
import builtins, inspect, os, functools

def _unwrap_until(fn, marker):
    seen = set()
    while getattr(fn, marker, False) and fn not in seen:
        seen.add(fn)
        fn = getattr(fn, "__wrapped__", getattr(fn, "_orig", fn))
    return fn

def _force_frames(self, n:int):
    for name in ('frame_num','num_frames','frames','length','video_frames','T'):
        try: setattr(self, name, int(n))
        except Exception: pass
    for cfg_name in ('config','cfg','conf'):
        c = getattr(self, cfg_name, None)
        if c is None: continue
        for name in ('frame_num','num_frames','frames','length','video_frames','T'):
            try: setattr(c, name, int(n))
            except Exception: pass

def _patch_wani2v(mod):
    try:
        cls = getattr(mod, "WanI2V", None)
        if cls is None:
            return

        # callable()
        if not hasattr(cls, "__call__"):
            def __call__(self, *a, **k):
                for name in ("generate","infer","forward","run","predict","sample"):
                    fn = getattr(self, name, None)
                    if callable(fn):
                        return fn(*a, **k)
                raise TypeError("WanI2V is not callable and no known generate-like method was found")
            cls.__call__ = __call__

        # shim для именованных аргументов
        gen = getattr(cls, "generate", None)
        if callable(gen) and not getattr(gen, "_shimmed_kwmap", False):
            sig = inspect.signature(gen); params = sig.parameters
            @functools.wraps(gen)
            def _shim(self, *args, **kwargs):
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
            _shim._shimmed_kwmap = True
            cls.generate = _shim

        # boundary OFF
        if os.environ.get("WAN_FORCE_BOUNDARY_OFF","0") == "1" and not getattr(cls, "_force_boundary_off", False):
            base = _unwrap_until(cls.generate, "_boundary_wrapped")
            @functools.wraps(base)
            def _wrap_boundary(self, *a, **k):
                try: self.boundary = 0
                except Exception: pass
                return base(self, *a, **k)
            _wrap_boundary._boundary_wrapped = True
            _wrap_boundary._orig = base
            cls.generate = _wrap_boundary
            cls._force_boundary_off = True

        # FRAMES hook
        if not getattr(cls, "_force_frames_hook", False):
            base0 = _unwrap_until(cls.generate, "_frames_wrapped")
            @functools.wraps(base0)
            def _wrap_frames(self, *a, **k):
                n = os.environ.get("WAN_FORCE_FRAMES")
                if n:
                    try: n = int(n)
                    except Exception: n = None
                if not n:
                    for key in ("frame_num","num_frames","frames","length"):
                        v = k.get(key)
                        if v is not None:
                            try: n = int(v); break
                            except Exception: pass
                if n:
                    _force_frames(self, n)
                    sig = inspect.signature(base0)
                    for key in ("frame_num","num_frames","frames","length"):
                        if key in sig.parameters:
                            k[key] = n; break
                    if "fps" in k and "seconds" not in k:
                        try:
                            fps = float(k["fps"])
                            if fps > 0: k["seconds"] = max(n / fps, 1.0 / fps)
                        except Exception:
                            pass
                return base0(self, *a, **k)
            _wrap_frames._frames_wrapped = True
            _wrap_frames._orig = base0
            cls.generate = _wrap_frames
            cls._force_frames_hook = True

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
print("[start] wrote sitecustomize.py (safe wraps: shim + boundary OFF + frames)")
PY_SITE


# -----------------------------------------------------------------------------
# Диагностика
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_INFO'
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
PY_INFO

# -----------------------------------------------------------------------------
# Гарантируем diffusers >= 0.35 (не даунгрейдим)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_DIF'
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
PY_DIF

# -----------------------------------------------------------------------------
# vae2_1.py: diffusers VAE backend + hard override _video_vae + guard .pth + encode unwrap
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_VAE'
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
        print("[patch] vae2_1.py: encode() unwrap applied")
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
            print("[patch] vae2_1.py: encode() unwrap applied (variant)")

    p.write_text(s, encoding="utf-8")
PY_VAE

# -----------------------------------------------------------------------------
# image2video.py: low/high_noise_checkpoint опциональны (ENV-aware)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_NOISE'
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
PY_NOISE

# -----------------------------------------------------------------------------
# image2video.py: глобальный безопасный патч boundary (normalize + compare helper)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_BOUNDARY'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    helpers = r"""
# --- safe boundary helpers (inserted) ---
def _norm_boundary(self):
    _b = getattr(self, 'boundary', None)
    _n = int(getattr(self, 'num_train_timesteps', 1000))
    if isinstance(_b, dict) or hasattr(_b, 'get'):
        if not _b.get('enabled', False):
            return None
        lo = int(max(0, min(1, float(_b.get('lower', 0.0)))) * _n)
        up = int(max(0, min(1, float(_b.get('upper', 1.0)))) * _n)
        return (lo, up)
    try:
        return int(float(_b) * _n)
    except Exception:
        return None

def _ge_boundary(t_item, boundary):
    if boundary is None:
        return False
    if isinstance(boundary, tuple):
        try:
            _, up = boundary
        except Exception:
            up = int(boundary[1]) if hasattr(boundary,'__len__') and len(boundary) > 1 else 0
        return t_item >= up
    try:
        return t_item >= int(boundary)
    except Exception:
        return False
# --- end helpers ---
"""
    if "_norm_boundary(" not in s:
        if "class WanI2V" in s:
            s = s.replace("class WanI2V", helpers + "\nclass WanI2V", 1)
        else:
            s = helpers + "\n" + s
        changed = True

    pat_mult = re.compile(r'^\s*boundary\s*=\s*self\.boundary\s*\*\s*self\.num_train_timesteps\s*$', re.M)
    if pat_mult.search(s):
        s = pat_mult.sub("boundary = _norm_boundary(self)", s); changed = True

    pat_tern = re.compile(
        r'^\s*sample_guide_scale\s*=\s*guide_scale\[\s*1\s*\]\s*if\s*t\.item\(\)\s*>=\s*boundary\s*else\s*guide_scale\[\s*0\s*\]\s*$',
        re.M
    )
    def repl_tern(m):
        ind = m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())]
        return f"{ind}_cond = _ge_boundary(t.item(), boundary)\n{ind}sample_guide_scale = guide_scale[1] if _cond else guide_scale[0]"
    s2 = pat_tern.sub(repl_tern, s)
    if s2 != s: s = s2; changed = True

    s2 = re.sub(r'if\s+t\.item\(\)\s*>=\s*boundary\s*:', r'if _ge_boundary(t.item(), boundary):', s)
    if s2 != s: s = s2; changed = True

    if "boundary = _norm_boundary(self)" not in s:
        s2 = re.sub(
            r"(?ms)^(\s*def\s+generate\s*\(.*?\):\s*\n)(\s*)(\S)",
            lambda m: f"{m.group(1)}{m.group(2)}boundary = _norm_boundary(self)\n{m.group(2)}{m.group(3)}",
            s
        )
        if s2 != s:
            s = s2; changed = True
        else:
            s2 = s.replace("for t in ", "boundary = _norm_boundary(self)\nfor t in ", 1)
            if s2 != s:
                s = s2; changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: boundary normalized + safe compare (global)")
    else:
        print("[patch] image2video.py: boundary already patched")
PY_BOUNDARY

# -----------------------------------------------------------------------------
# ДОП. патч: многострочные варианты сравнения/тернарника с boundary
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_BOUNDARY_MULTILINE'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] not found:", p)
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    pat_tern_ml = re.compile(
        r'(?P<ind>^[ \t]*)sample_guide_scale\s*=\s*guide_scale\[\s*1\s*\]\s*'
        r'if\s*t\.item\(\s*[\s\S]*?\)\s*>=\s*boundary\s*else\s*guide_scale\[\s*0\s*\][ \t]*$',
        re.M
    )
    def repl_tern_ml(m):
        ind = m.group('ind')
        return (f"{ind}_cond = _ge_boundary(t.item(), boundary)\n"
                f"{ind}sample_guide_scale = guide_scale[1] if _cond else guide_scale[0]")
    s2 = pat_tern_ml.sub(repl_tern_ml, s)
    if s2 != s:
        s = s2; changed = True

    pat_if_ml = re.compile(
        r'(?P<ind>^[ \t]*)if\s*t\.item\(\s*[\s\S]*?\)\s*>=\s*boundary\s*:[ \t]*$',
        re.M
    )
    def repl_if_ml(m):
        ind = m.group('ind')
        return f"{ind}if _ge_boundary(t.item(), boundary):"
    s2 = pat_if_ml.sub(repl_if_ml, s)
    if s2 != s:
        s = s2; changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: multiline boundary fixes applied")
    else:
        print("[patch] image2video.py: multiline patterns not found (maybe already patched)")
PY_BOUNDARY_MULTILINE

# -----------------------------------------------------------------------------
# image2video.py: MASK FIX — выровнять число каналов маски до кратного 4
# и синхронизировать frame_num, прежде чем делать view(... //4, 4, ...)
# (ремейк с сохранением исходного отступа)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_MASK_FIX'
from pathlib import Path
import re

p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] image2video.py not found for mask-fix")
else:
    s = p.read_text(encoding="utf-8")
    # Паттерн: захватываем ведущий отступ строки с view(...)
    pat = re.compile(
        r"(?P<ind>^[ \t]*)msk\s*=\s*msk\.view\(\s*1\s*,\s*msk\.shape\[1\]\s*//\s*4\s*,\s*4\s*,\s*lat_h\s*,\s*lat_w\s*\)",
        re.M
    )

    def repl(m):
        ind = m.group("ind")
        return (
            f"{ind}##__MASK_CHANNEL_GUARD__\n"
            f"{ind}_C = int(msk.shape[1])\n"
            f"{ind}if (_C % 4) != 0:\n"
            f"{ind}    _C_trim = _C - (_C % 4)\n"
            f'{ind}    print(f"[mask] trim channels: {{_C}} -> {{_C_trim}} (not divisible by 4)")\n'
            f"{ind}    msk = msk[:, :_C_trim, ...]\n"
            f"{ind}    _C = _C_trim\n"
            f"{ind}try:\n"
            f"{ind}    self.frame_num = int(_C // 4)\n"
            f"{ind}except Exception:\n"
            f"{ind}    pass\n"
            f"{ind}msk = msk.view(1, _C // 4, 4, lat_h, lat_w)"
        )

    changed = False
    if "##__MASK_CHANNEL_GUARD__" in s:
        # Нормализуем уже вставленный блок по его текущему отступу (строка с маркером задаёт ind)
        guard_pat = re.compile(r"(?P<ind>^[ \t]*)##__MASK_CHANNEL_GUARD__\s*$", re.M)
        m = guard_pat.search(s)
        if m:
            ind = m.group("ind")
            # Пересоберём блок до следующего view(...), на всякий
            block_pat = re.compile(
                r"^[ \t]*##__MASK_CHANNEL_GUARD__.*?msk\s*=\s*msk\.view\(1,\s*_C\s*//\s*4\s*,\s*4\s*,\s*lat_h\s*,\s*lat_w\)",
                re.M | re.S
            )
            s = block_pat.sub(
                (
                    f"{ind}##__MASK_CHANNEL_GUARD__\n"
                    f"{ind}_C = int(msk.shape[1])\n"
                    f"{ind}if (_C % 4) != 0:\n"
                    f"{ind}    _C_trim = _C - (_C % 4)\n"
                    f'{ind}    print(f"[mask] trim channels: {{_C}} -> {{_C_trim}} (not divisible by 4)")\n'
                    f"{ind}    msk = msk[:, :_C_trim, ...]\n"
                    f"{ind}    _C = _C_trim\n"
                    f"{ind}try:\n"
                    f"{ind}    self.frame_num = int(_C // 4)\n"
                    f"{ind}except Exception:\n"
                    f"{ind}    pass\n"
                    f"{ind}msk = msk.view(1, _C // 4, 4, lat_h, lat_w)"
                ),
                s,
                count=1
            )
            changed = True
    else:
        s2 = pat.sub(repl, s, count=1)
        if s2 != s:
            s = s2
            changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: mask channels guard inserted/re-indented")
    else:
        print("[patch] image2video.py: mask guard unchanged (pattern not found)")
PY_MASK_FIX

# -----------------------------------------------------------------------------
# model.py: CHANNEL GUARD — привести каналы к 4*frame_num (напр. 48 при 12 кадрах)
# до patch_embedding, чтобы Conv3d(48, …) не падал на 68 каналах
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_CHAN_GUARD'
from pathlib import Path
import re

p = Path("/app/Wan2.2/wan/modules/model.py")
if not p.exists():
    print("[patch][warn] model.py not found for channel-guard")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) Вставим helper (один раз, идемпотентно)
    if "_ensure_frame_channels(" not in s:
        helper = r"""

# --- inserted: ensure frame channels match 4*frame_num ---
def _ensure_frame_channels(self, u):
    # u: Tensor [C, H, W] (стек каналов латента)
    import torch
    C = int(u.shape[0])
    want = None
    for src in (getattr(self, "frame_num", None),
                getattr(getattr(self, "config", None), "frame_num", None),
                getattr(getattr(self, "config", None), "T", None)):
        try:
            if src is not None:
                want = int(src)
                break
        except Exception:
            pass
    if not want or want <= 0:
        want = 12
    targetC = 4 * want
    if C == targetC:
        return u
    if C > targetC:
        print(f"[chan] trim channels: {C} -> {targetC}")
        return u[:targetC, ...]
    # C < targetC: дополним последним каналом
    rep = targetC - C
    return torch.cat([u, u[-1:, ...].repeat(rep, 1, 1)], dim=0)
# --- end inserted ---
"""
        idx = s.find("class ")
        if idx != -1:
            s = s[:idx] + helper + s[idx:]
            changed = True

    # 2) Перед patch_embedding пропустим через гард
    # x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    pat = re.compile(r"x\s*=\s*\[self\.patch_embedding\(\s*u\.unsqueeze\(0\)\s*\)\s*for\s+u\s+in\s+x\s*\]")
    if pat.search(s) and "_ensure_frame_channels(self, u).unsqueeze(0)" not in s:
        s = pat.sub("x = [self.patch_embedding(_ensure_frame_channels(self, u).unsqueeze(0)) for u in x]", s)
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] model.py: channel guard installed")
    else:
        print("[patch] model.py: channel guard already present or pattern not found")
PY_CHAN_GUARD

# -----------------------------------------------------------------------------
# attention.py: SDPA fallback (фикс отступа внутри ветки else)
#   - заменяем 'assert FLASH_ATTN_2_AVAILABLE' на корректно ИНДЕНТИРОВАННЫЙ fallback
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_FALLBACK_FIX'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/modules/attention.py")
if not p.exists():
    print("[patch][warn] attention.py not found for SDPA fallback fix")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) Вариант без комментария
    pat = re.compile(r"(?m)^(?P<ind>[ \t]*)assert[ \t]+FLASH_ATTN_2_AVAILABLE[ \t]*$")
    def repl(m):
        ind = m.group("ind")
        return (
            f"{ind}##__SDPA_FALLBACK__\n"
            f"{ind}if not FLASH_ATTN_2_AVAILABLE:\n"
            f"{ind}    import torch.nn.functional as F\n"
            f"{ind}    # F.scaled_dot_product_attention(q,k,v,attn_mask,dropout,is_causal)\n"
            f"{ind}    return F.scaled_dot_product_attention(q, k, v, None, dropout_p, is_causal)"
        )
    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2; changed = True

    # 2) Вариант с комментарием в конце assert-строки
    pat2 = re.compile(r"(?m)^(?P<ind>[ \t]*)assert[ \t]+FLASH_ATTN_2_AVAILABLE[^\n]*$")
    s2 = pat2.sub(repl, s)
    if s2 != s:
        s = s2; changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] attention.py: SDPA fallback re-indented")
    else:
        print("[patch] attention.py: fallback already fixed or assert not found")
PY_ATTN_FALLBACK_FIX

# -----------------------------------------------------------------------------
# attention.py: SAFE SDPA WRAPPER — выровнять dtype/device и ручной fallback
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_SAFE'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/modules/attention.py")
if not p.exists():
    print("[patch][warn] attention.py not found for safe SDPA wrapper")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # Вставим helper один раз
    if "def _wan_sdpa_safe(" not in s:
        helper = r"""

# === inserted: safe SDPA wrapper ===
def _wan_sdpa_safe(q, k, v, dropout_p=0.0, is_causal=False):
    import torch, torch.nn.functional as F
    dev = q.device
    # Выравниваем устройство
    if k.device != dev: k = k.to(dev)
    if v.device != dev: v = v.to(dev)
    # Выравниваем тип: на CUDA -> fp16, иначе fp32
    want = torch.float16 if dev.type == "cuda" else torch.float32
    if q.dtype not in (torch.float16, torch.float32): q = q.to(want)
    if k.dtype not in (torch.float16, torch.float32): k = k.to(want)
    if v.dtype not in (torch.float16, torch.float32): v = v.to(want)
    if dev.type == "cuda":
        q = q.to(torch.float16); k = k.to(torch.float16); v = v.to(torch.float16)
    else:
        q = q.to(torch.float32); k = k.to(torch.float32); v = v.to(torch.float32)
    # Континуальность
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    # Dropout в инференсе = 0
    try:
        if not torch.is_grad_enabled(): dropout_p = 0.0
    except Exception:
        dropout_p = 0.0
    # Основной путь: SDPA
    try:
        return F.scaled_dot_product_attention(q, k, v, None, dropout_p, is_causal)
    except Exception as e:
        # Ручной бэкап: softmax(QK^T/sqrt(d)) V
        try:
            d = q.shape[-1]
            scale = (1.0 / (d ** 0.5))
            attn = (q @ k.transpose(-2, -1)) * scale
            if is_causal:
                mask = torch.triu(torch.ones_like(attn), diagonal=1)
                attn = attn.masked_fill(mask.bool(), float("-inf"))
            attn = attn.softmax(dim=-1)
            return attn @ v
        except Exception as e2:
            raise RuntimeError(f"wan-sdpa failed: {e} ; fallback failed: {e2}")
# === end inserted ===
"""
        # Вставим helper перед первой декларацией класса/функции
        idx = s.find("class ")
        if idx == -1:
            idx = s.find("def ")
        if idx != -1:
            s = s[:idx] + helper + s[idx:]
            changed = True

    # Подменим место, где мы ранее ставили SDPA fallback (метка ##__SDPA_FALLBACK__)
    pat = re.compile(r"(?m)^[ \t]*##__SDPA_FALLBACK__\s*\n[ \t]*if not FLASH_ATTN_2_AVAILABLE:[\s\S]*?return[^\n]*$")
    def repl(m):
        # Сохраняем отступ первой строки блока
        first_line = m.group(0).splitlines()[0]
        ind = first_line[:len(first_line)-len(first_line.lstrip())]
        return (
            f"{ind}##__SDPA_FALLBACK__\n"
            f"{ind}if not FLASH_ATTN_2_AVAILABLE:\n"
            f"{ind}    return _wan_sdpa_safe(q, k, v, dropout_p, is_causal)"
        )
    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] attention.py: safe SDPA wrapper installed")
    else:
        print("[patch] attention.py: already safe")
PY_ATTN_SAFE

# -----------------------------------------------------------------------------
# attention.py: fix NameError(is_causal) — всегда некаузально для WanSelfAttention
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_ISCAUSAL_FIX'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/modules/attention.py")
if not p.exists():
    print("[patch][warn] attention.py not found for is_causal fix")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) Наш безопасный хелпер мог вызываться с is_causal -> заменим на False
    s2 = re.sub(
        r"_wan_sdpa_safe\s*\(\s*q\s*,\s*k\s*,\s*v\s*,\s*dropout_p\s*,\s*is_causal\s*\)",
        "_wan_sdpa_safe(q, k, v, dropout_p, False)",
        s
    )
    if s2 != s:
        s = s2; changed = True

    # 2) Прямой SDPA мог вызываться с is_causal -> заменим на False
    s2 = re.sub(
        r"scaled_dot_product_attention\s*\(\s*q\s*,\s*k\s*,\s*v\s*,\s*None\s*,\s*dropout_p\s*,\s*is_causal\s*\)",
        "scaled_dot_product_attention(q, k, v, None, dropout_p, False)",
        s
    )
    if s2 != s:
        s = s2; changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] attention.py: is_causal -> False (non-causal SDPA)")
    else:
        print("[patch] attention.py: nothing to change (already non-causal)")
PY_ATTN_ISCAUSAL_FIX

# -----------------------------------------------------------------------------
# attention.py: SDPA merge heads -> (B, T, H*D) to match Linear(3072,3072)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_RETURN_SHAPE'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/modules/attention.py")
if not p.exists():
    print("[patch][warn] attention.py not found for return-shape fix")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) Вставим helper для слияния голов (один раз)
    if "def _wan_sdpa_merge(" not in s:
        helper = r"""

# === inserted: sdpa merge to (B,T,H*D) ===
def _wan_sdpa_merge(q, k, v, dropout_p=0.0, is_causal=False):
    # q,k,v ожидаются как (B,H,T,D)
    out = _wan_sdpa_safe(q, k, v, dropout_p, False)  # non-causal
    B, H, T, D = out.shape
    # (B,H,T,D) -> (B,T,H,D) -> (B,T,H*D)
    out = out.transpose(1, 2).contiguous().view(B, T, H * D)
    return out
# === end inserted ===
"""
        # Вставим helper перед первой декларацией класса/функции
        idx = s.find("class ")
        if idx == -1:
            idx = s.find("def ")
        if idx != -1:
            s = s[:idx] + helper + s[idx:]
            changed = True

    # 2) Заменим блок нашего фолбэка, чтобы возвращать слитую форму
    pat = re.compile(
        r"(?ms)^([ \t]*)##__SDPA_FALLBACK__\s*\n\1if not FLASH_ATTN_2_AVAILABLE:\s*\n.*?return[^\n]*$"
    )
    def repl(m):
        ind = m.group(1)
        return (
            f"{ind}##__SDPA_FALLBACK__\n"
            f"{ind}if not FLASH_ATTN_2_AVAILABLE:\n"
            f"{ind}    return _wan_sdpa_merge(q, k, v, dropout_p, False)"
        )
    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] attention.py: SDPA now returns (B,T,H*D)")
    else:
        print("[patch] attention.py: return-shape already correct or markers not found")
PY_ATTN_RETURN_SHAPE

# -----------------------------------------------------------------------------
# attention.py: robust SDPA merge (поддержка 3D и 4D выходов)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_RETURN_SHAPE_V2'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/modules/attention.py")
if not p.exists():
    print("[patch][warn] attention.py not found for robust return-shape")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # Переписываем тело _wan_sdpa_merge на робастное (3D/4D)
    pat = re.compile(
        r"(?ms)^def\s+_wan_sdpa_merge\s*\(\s*q\s*,\s*k\s*,\s*v\s*,\s*dropout_p\s*=\s*0\.0\s*,\s*is_causal\s*=\s*False\s*\)\s*:\s*.*?# === end inserted ==="
    )
    new_body = r"""def _wan_sdpa_merge(q, k, v, dropout_p=0.0, is_causal=False):
    # Стремимся вернуть (B,T,C). Если SDPA дал (B,H,T,D) — сливаем головы.
    out = _wan_sdpa_safe(q, k, v, dropout_p, False)  # non-causal
    if out.dim() == 4:
        B, H, T, D = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return out
    elif out.dim() == 3:
        # Уже (B,T,C): отдаём как есть
        return out
    else:
        raise RuntimeError(f"unexpected SDPA output rank: {out.dim()} with shape {tuple(out.shape)}")
# === end inserted ==="""
    s2 = pat.sub(new_body, s)
    if s2 != s:
        s = s2
        changed = True
    else:
        # если сигнатура отличается — просто вставим/заменим определение функции
        if "def _wan_sdpa_merge(" in s:
            s = re.sub(r"(?ms)^def\s+_wan_sdpa_merge\(.*?\)\s*:\s*.*?$", new_body, s)
            changed = True
        else:
            # не нашли — вставим helper перед первой декларацией класса/функции
            idx = s.find("class ")
            if idx == -1:
                idx = s.find("def ")
            if idx != -1:
                s = s[:idx] + "\n\n" + new_body + "\n\n" + s[idx:]
                changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] attention.py: robust SDPA merge installed")
    else:
        print("[patch] attention.py: merge already robust")
PY_ATTN_RETURN_SHAPE_V2

# -----------------------------------------------------------------------------
# attention.py: WAN_FORCE_EAGER_ATTN=1 → всегда ручной путь (без SDPA/Flash)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_EAGER'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/modules/attention.py")
if not p.exists():
    print("[patch][warn] attention.py not found for eager-attn switch")
else:
    s = p.read_text(encoding="utf-8")
    if "def _wan_sdpa_safe(" in s and "WAN_FORCE_EAGER_ATTN" not in s:
        s = s.replace(
            "def _wan_sdpa_safe(q, k, v, dropout_p=0.0, is_causal=False):",
            "def _wan_sdpa_safe(q, k, v, dropout_p=0.0, is_causal=False):\n    import os as _os"
        ).replace(
            "    # Основной путь: SDPA",
            "    # Принудительный обход SDPA\n"
            "    if _os.environ.get('WAN_FORCE_EAGER_ATTN','0') == '1':\n"
            "        pass  # сразу пойдём в ручной бэкап ниже\n\n"
            "    # Основной путь: SDPA"
        )
        Path(p).write_text(s, encoding="utf-8")
        print("[patch] attention.py: eager-attn switch added")
    else:
        print("[patch] attention.py: switch already present or safe wrapper missing")
PY_ATTN_EAGER

# -----------------------------------------------------------------------------
# model.py: guard перед self.o(x) — слияние голов до in_features (например 3072)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_ATTN_O_GUARD'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/modules/model.py")
if not p.exists():
    print("[patch][warn] model.py not found for attn-o guard")
else:
    s = p.read_text(encoding="utf-8")
    # Ищем строку "x = self.o(x)" в WanSelfAttention.forward (лог указывает на ~строку 183)
    pat = re.compile(r"(?m)^(\s*)x\s*=\s*self\.o\(x\)\s*$")
    def repl(m):
        ind = m.group(1)
        return (
f"""{ind}# __ATTN_O_GUARD__: если last-dim < in_features и кратен — восстановим H и сольём головы
{ind}_x_in = x
{ind}try:
{ind}    _exp = getattr(self.o, 'in_features', None)
{ind}    _d   = int(_x_in.shape[-1])
{ind}    if _exp and _d != _exp and (_exp % _d) == 0:
{ind}        _H   = _exp // _d
{ind}        _n0  = _x_in.numel() // _d
{ind}        if (_n0 % _H) == 0:
{ind}            _x_in = _x_in.reshape(_n0//_H, _H, _d).reshape(_n0//_H, _exp)
{ind}    x = self.o(_x_in)
{ind}except Exception:
{ind}    x = self.o(x)"""
        )
    s2, n = pat.subn(repl, s, count=1)
    if n > 0:
        p.write_text(s2, encoding="utf-8")
        print("[patch] model.py: self.o() input guard installed")
    else:
        print("[patch] model.py: pattern 'x = self.o(x)' not found (maybe already patched?)")
PY_ATTN_O_GUARD

# -----------------------------------------------------------------------------
# model.py: SKIP cross-attn по env WAN_FORCE_NO_TEXT=1 или пустому промпту
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_SKIP_XATTN'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/modules/model.py")
if not p.exists():
    print("[patch][warn] model.py not found for skip-xattn")
else:
    s = p.read_text(encoding="utf-8")
    # Меняем строку: x = x + self.cross_attn(self.norm3(x), context, context_lens)
    pat = re.compile(r"(?m)^(\s*)x\s*=\s*x\s*\+\s*self\.cross_attn\(\s*self\.norm3\(x\)\s*,\s*context\s*,\s*context_lens\s*\)\s*$")
    def repl(m):
        ind = m.group(1)
        return (
f"""{ind}# __SKIP_XATTN__: по флагу окольцовываем без cross-attn
{ind}import os as _os
{ind}_skip = _os.environ.get("WAN_FORCE_NO_TEXT","0") == "1"
{ind}if (context is None) or _skip:
{ind}    x = x  # пропускаем cross-attn
{ind}else:
{ind}    x = x + self.cross_attn(self.norm3(x), context, context_lens)"""
        )
    s2, n = pat.subn(repl, s, count=1)
    if n > 0:
        p.write_text(s2, encoding="utf-8")
        print("[patch] model.py: SKIP_XATTN installed")
    else:
        print("[patch] model.py: pattern for SKIP_XATTN not found (maybe already patched?)")
PY_SKIP_XATTN

# -----------------------------------------------------------------------------
# model.py: CROSS-BATCH ALIGN в WanCrossAttention.forward (Q vs K/V)
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_CROSS_BATCH_ALIGN'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/modules/model.py")
if not p.exists():
    print("[patch][warn] model.py not found for cross-batch align")
else:
    s = p.read_text(encoding="utf-8")
    # Вариант А: заменяем сам вызов flash_attention(...)
    patA = re.compile(r"(?m)^(\s*)x\s*=\s*flash_attention\(\s*q\s*,\s*k\s*,\s*v\s*,\s*k_lens\s*=\s*context_lens\s*\)\s*$")
    def replA(m):
        ind = m.group(1)
        return (
f"""{ind}# __CROSS_BATCH_ALIGN__: выровнять batch-ось Q vs K/V
{ind}if q.shape[0] != k.shape[0]:
{ind}    import torch
{ind}    bq, bk = q.shape[0], k.shape[0]
{ind}    if bk == 1:
{ind}        k = k.expand(bq, -1, -1)
{ind}        v = v.expand(bq, -1, -1)
{ind}    elif bq % bk == 0:
{ind}        rep = bq // bk
{ind}        k = torch.repeat_interleave(k, rep, dim=0)
{ind}        v = torch.repeat_interleave(v, rep, dim=0)
{ind}x = flash_attention(q, k, v, k_lens=context_lens)"""
        )
    s2, nA = patA.subn(replA, s, count=1)
    if nA == 0:
        # Вариант B: вставка перед flash_attention(...) внутри WanCrossAttention.forward
        # Найдём функцию forward класса WanCrossAttention
        patHead = re.compile(r"(?ms)^class\s+WanCrossAttention\b.*?^\s*def\s+forward\s*\(\s*self\s*,\s*x\s*,\s*context\s*,\s*context_lens\s*\)\s*:\s*\n")
        mHead = patHead.search(s)
        if mHead:
            insert_pos = mHead.end()
            # Вставим блок выравнивания прямо перед первым вызовом flash_attention
            patCall = re.compile(r"(?m)^(\s*)x\s*=\s*flash_attention\(")
            mCall = patCall.search(s, pos=insert_pos)
            if mCall:
                ind = mCall.group(1)
                guard = (
f"""{ind}# __CROSS_BATCH_ALIGN__: выровнять batch-ось Q vs K/V (in-forward)
{ind}if q.shape[0] != k.shape[0]:
{ind}    import torch
{ind}    bq, bk = q.shape[0], k.shape[0]
{ind}    if bk == 1:
{ind}        k = k.expand(bq, -1, -1)
{ind}        v = v.expand(bq, -1, -1)
{ind}    elif bq % bk == 0:
{ind}        rep = bq // bk
{ind}        k = torch.repeat_interleave(k, rep, dim=0)
{ind}        v = torch.repeat_interleave(v, rep, dim=0)
"""
                )
                s2 = s[:mCall.start()] + guard + s[mCall.start():]
                nA = 1
            else:
                s2 = s
        else:
            s2 = s
    if s2 != s and nA > 0:
        p.write_text(s2, encoding="utf-8")
        print("[patch] model.py: cross-attn batch align installed")
    else:
        print("[patch] model.py: could not install cross-batch align (pattern mismatch?)")
PY_CROSS_BATCH_ALIGN

# -----------------------------------------------------------------------------
# image2video.py: принудительно выровнять каналы sample к 4 * frame_num
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_LATENT_CHAN_SYNC'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[patch][warn] image2video.py not found for latent-chan sync")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) Вставим helper один раз
    if "_ensure_latent_channels(" not in s:
        helper = r"""
# --- inserted: ensure latent channels match 4*frame_num ---
def _ensure_latent_channels(sample, frame_num:int):
    import torch
    want = int(frame_num) * 4
    if not hasattr(sample, "shape"): 
        return sample
    if sample.dim() < 2: 
        return sample
    have = int(sample.shape[1])
    if have == want:
        return sample
    # trim or pad with fresh noise on the same device/dtype/shape tail
    B = int(sample.shape[0])
    tail = sample.shape[2:]
    if have > want:
        return sample[:, :want, ...]
    # pad
    pad = torch.randn((B, want - have, *tail), device=sample.device, dtype=sample.dtype)
    return torch.cat([sample, pad], dim=1)
# --- end inserted ---
"""
        # Вставим перед первой декларацией класса/функции
        idx = s.find("class ")
        if idx == -1:
            idx = s.find("def ")
        if idx != -1:
            s = s[:idx] + helper + s[idx:]
            changed = True

    # 2) Перед самым первым вызовом scheduler.step(...) гарантируем выравнивание
    # Ищем первую строку "temp_x0 = sample_scheduler.step("
    pat = re.compile(r"(?m)^(\s*)temp_x0\s*=\s*sample_scheduler\.step\(")
    m = pat.search(s)
    if m and "_ENSURE_LATENT_CHANS_APPLIED__" not in s:
        ind = m.group(1)
        inject = (
            f"{ind}# __ENSURE_LATENT_CHANS_APPLIED__\n"
            f"{ind}try:\n"
            f"{ind}    sample = _ensure_latent_channels(sample, getattr(self, 'frame_num', 1))\n"
            f"{ind}except Exception:\n"
            f"{ind}    pass\n"
        )
        s = s[:m.start()] + inject + s[m.start():]
        changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] image2video.py: latent channel sync before scheduler.step")
    else:
        print("[patch] image2video.py: latent chan sync already present or pattern not found")
PY_LATENT_CHAN_SYNC

# -----------------------------------------------------------------------------
# fm_solvers_unipc.py: выровнять каналы sample к model_output в convert_model_output
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_SCHED_CHAN_GUARD'
from pathlib import Path, re
p = Path("/app/Wan2.2/wan/utils/fm_solvers_unipc.py")
if not p.exists():
    print("[patch][warn] fm_solvers_unipc.py not found for scheduler channel guard")
else:
    s = p.read_text(encoding="utf-8")
    changed = False

    # 1) helper один раз
    if "_align_channels_like(" not in s:
        helper = r"""
# --- inserted: align channels of A to B (trim/pad with noise) ---
def _align_channels_like(a, b):
    import torch
    if not hasattr(a, "shape") or not hasattr(b, "shape"):
        return a
    if a.dim() < 2 or b.dim() < 2:
        return a
    ca = int(a.shape[1]); cb = int(b.shape[1])
    if ca == cb:
        return a
    tail = a.shape[2:]
    if ca > cb:
        return a[:, :cb, ...]
    # pad with fresh noise matching device/dtype
    pad = torch.randn((a.shape[0], cb - ca, *tail), device=a.device, dtype=a.dtype)
    return torch.cat([a, pad], dim=1)
# --- end inserted ---
"""
        # вставим перед первой функцией
        idx = s.find("def ")
        if idx == -1:
            idx = len(s)
        s = s[:idx] + helper + s[idx:]
        changed = True

    # 2) в convert_model_output, перед x0_pred = sample - sigma_t * model_output
    pat = re.compile(
        r"(?P<ind>^[ \t]*)x0_pred\s*=\s*sample\s*-\s*sigma_t\s*\*\s*model_output\s*$",
        re.M
    )
    def repl(m):
        ind = m.group("ind")
        return (
f"""{ind}# __CHAN_GUARD__: align sample channels to model_output
{ind}sample = _align_channels_like(sample, model_output)
{ind}x0_pred = sample - sigma_t * model_output"""
        )
    s2, n = pat.subn(repl, s, count=1)
    if n > 0:
        s = s2; changed = True

    if changed:
        p.write_text(s, encoding="utf-8")
        print("[patch] fm_solvers_unipc.py: convert_model_output channel guard installed")
    else:
        print("[patch] fm_solvers_unipc.py: channel guard already present or pattern not found")
PY_SCHED_CHAN_GUARD

# -----------------------------------------------------------------------------
# re-indent: если boundary-вставка стоит сразу после `with …:` — добавить нужный отступ
# -----------------------------------------------------------------------------
${PYBIN} - <<'PY_REINDENT'
from pathlib import Path
p = Path("/app/Wan2.2/wan/image2video.py")
if not p.exists():
    print("[fix][warn] image2video.py not found for reindent")
else:
    lines = p.read_text(encoding="utf-8").splitlines()

    def ws_of(s: str):
        return s[:len(s)-len(s.lstrip())]

    def indent_unit(sample_ws: str):
        return '\t' if (sample_ws and sample_ws[0] == '\t') else ' ' * 4

    changed = False
    for i, line in enumerate(lines):
        if line.strip() == "boundary = _norm_boundary(self)":
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j >= 0 and lines[j].rstrip().endswith(':'):
                base_ws = ws_of(lines[j])
                want_ws = base_ws + indent_unit(base_ws)
                have_ws = ws_of(line)
                if have_ws != want_ws:
                    lines[i] = f"{want_ws}boundary = _norm_boundary(self)"
                    changed = True

    if changed:
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("[fix] boundary line re-indented under its enclosing ':' block")
    else:
        print("[fix] nothing to change (already OK or not found)")
PY_REINDENT

# -----------------------------------------------------------------------------
# Запуск хэндлера
# -----------------------------------------------------------------------------
echo "[start] Launching handler..."
exec ${PYBIN} -u /app/handler.py
