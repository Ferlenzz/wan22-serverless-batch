#!/usr/bin/env bash
set -e

python3 - <<'PY'
import os, torch
try:
    import diffusers; dv = diffusers.__version__
except Exception as e:
    dv = f"<not-importable: {e}>"
try:
    import transformers; tv = transformers.__version__
except Exception as e:
    tv = f"<not-importable: {e}>"

print("[runtime] torch", torch.__version__)
print("[runtime] diffusers", dv)
print("[runtime] transformers", tv)
print("[runtime] USE_DIFFUSERS_VAE =", os.environ.get("USE_DIFFUSERS_VAE"))
PY

exec python3 -u /app/handler.py
