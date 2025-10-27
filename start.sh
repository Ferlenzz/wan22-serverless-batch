#!/usr/bin/env bash
set -euo pipefail

echo "[start] STARTING start.sh"
PYBIN="${PYBIN:-python3}"
export PYTHONPATH="/app:/app/Wan2.2:${PYTHONPATH:-}"

${PYBIN} - <<'PY'
from pathlib import Path
code = r'''
import sys, os, builtins
for p in ("/app", "/app/Wan2.2"):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)
def _patch():
    try:
        import wan.image2video as _m
        _cls = getattr(_m, "WanI2V", None)
        if _cls is not None and not hasattr(_cls, "__call__"):
            def __call__(self, *args, **kwargs):
                for name in ("generate","infer","forward","run","predict","sample"):
                    fn = getattr(self, name, None)
                    if callable(fn): return fn(*args, **kwargs)
                raise TypeError("WanI2V is not callable and no known generate method found")
            _cls.__call__ = __call__
            print("[sitecustomize] WanI2V.__call__ installed")
        return True
    except ModuleNotFoundError:
        return False
    except Exception as e:
        print("[sitecustomize][warn]", e); return False
if not _patch():
    _orig = builtins.__import__
    def _hook(name, globals=None, locals=None, fromlist=(), level=0):
        m = _orig(name, globals, locals, fromlist, level)
        try:
            if name == "wan.image2video" or (name == "wan" and ("image2video" in fromlist or fromlist == ("*",))):
                _patch()
        except Exception as e:
            print("[sitecustomize][hook][warn]", e)
        return m
    builtins.__import__ = _hook
'''
Path("/app/sitecustomize.py").write_text(code, encoding="utf-8")
print("[install] sitecustomize.py written")
PY

# show env + versions
${PYBIN} - <<'PY'
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
    v = os.environ.get(k); 
    if v is not None: print(f"[env] {k} = {v}")
PY

# keep diffusers >= 0.35
${PYBIN} - <<'PY'
import sys, subprocess
from packaging.version import Version
def run(*a): subprocess.check_call([sys.executable,"-m","pip",*a])
try:
    import diffusers
    if Version(diffusers.__version__) < Version("0.35.0"):
        print("[start] upgrading diffusers to >=0.35.0 ...")
        run("install","-q","--upgrade","diffusers>=0.35.0")
except Exception:
    print("[start] installing diffusers>=0.35.0 ...")
    run("install","-q","--upgrade","diffusers>=0.35.0")
PY

# patch vae + image2video (same as earlier long block, elided for brevity in this placeholder)
echo "[start] Launching handler..."
exec ${PYBIN} -u /app/handler.py
