# ---------- BASE ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- SYSTEM ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git curl ffmpeg libglib2.0-0 libgl1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------- TORCH (cu121) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# XFormers wheel (no-compile) for cu121
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# ---------- WAN 2.2 code ----------
WORKDIR /app
ARG WAN_REF=main
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2 && \
    cd /app/Wan2.2 && git fetch --all --tags && git checkout ${WAN_REF}

# Прячем flash-attn/xformers из reqs, чтобы ничего не компилилось
WORKDIR /app/Wan2.2
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt && \
    pip install --no-cache-dir --prefer-binary -r /tmp/req-pruned.txt

# ---------- Extra deps (runtime) ----------
# Разбиваем на 2 шага, чтобы проще диагностировать, и используем --prefer-binary
RUN pip install --no-cache-dir --prefer-binary \
      runpod==1.6.2 loguru==0.7.2 pillow==10.4.0 \
      imageio==2.36.0 imageio-ffmpeg==0.5.1 numpy==1.26.4 \
      decord==0.6.0 opencv-python-headless==4.10.0.84 \
      safetensors==0.4.5 einops==0.7.0 huggingface_hub==0.20.3

RUN pip install --no-cache-dir --prefer-binary \
      diffusers==0.30.2 transformers==4.44.2 accelerate==0.34.2 \
      peft==0.17.1

# ---------- ПАТЧ: фильтрованная загрузка VAE ----------
RUN python3 - <<'PY'
from pathlib import Path
import re
p = Path("/app/Wan2.2/wan/modules/vae2_1.py")
s = p.read_text(encoding="utf-8")
helper = r'''
def _load_filtered_state_dict(model, ckpt):
    ms = model.state_dict()
    ok, mism = {}, []
    for k, v in ckpt.items():
        if k in ms and getattr(v, "shape", None) == getattr(ms[k], "shape", None):
            ok[k] = v
        else:
            mism.append(k)
    missing = [k for k in ms.keys() if k not in ok]
    model.load_state_dict(ok, strict=False)
    print(f"[VAE] filtered load: loaded={len(ok)} missing={len(missing)} skipped(mismatch)={len(mism)}")
    return missing, mism
'''
if "def _load_filtered_state_dict" not in s:
    s = s.replace("import torch", "import torch\n" + helper, 1)
s = re.sub(
    r"model\.load_state_dict\(\s*torch\.load\(([^)]+)\)\s*,\s*assign=True\s*\)",
    r"missing, mism = _load_filtered_state_dict(model, torch.load(\1))",
    s, count=1
)
s = re.sub(
    r"model\.load_state_dict\(\s*torch\.load\(([^)]+)\)\s*\)",
    r"missing, mism = _load_filtered_state_dict(model, torch.load(\1))",
    s
)
p.write_text(s, encoding="utf-8")
print("patched", p)
PY

# ---------- APP ----------
WORKDIR /app
COPY engine.py /app/engine.py
COPY handler.py /app/handler.py

# ---------- ENV ----------
ENV RP_VOLUME=/runpod-volume
ENV WAN_ROOT=/app/Wan2.2
ENV PYTHONPATH=/app/Wan2.2:$PYTHONPATH
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HF_ENABLE_HF_TRANSFER=
ENV HF_HUB_ENABLE_HF_TRANSFER=

# ---------- ENTRY ----------
CMD ["python3","-u","/app/handler.py"]
