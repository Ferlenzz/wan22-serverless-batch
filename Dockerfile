# ---------- BASE ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- SYSTEM ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git ffmpeg libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------- TORCH (cu121) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# XFormers (готовое колесо под cu121). FLASH-ATTN не ставим.
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# ---------- WAN2.2 repo (для утилит и конфигов, зависимости без flash-attn/xformers) ----------
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2
WORKDIR /app/Wan2.2
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt
RUN pip install --no-cache-dir -r /tmp/req-pruned.txt

# ---------- COMFYUI + WAN WRAPPER ----------
WORKDIR /app
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI
RUN pip install --no-cache-dir -r /app/ComfyUI/requirements.txt
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git /app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper

# ---------- OUR CODE ----------
WORKDIR /app
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# ---------- RUNPOD SDK + LOGGING ----------
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
python3 -m pip install --no-cache-dir \
  "pydantic>=2,<3" \
  "typing-extensions>=4.13.0" \
  "annotated-types>=0.6" \
  runpod==1.7.13 \
  loguru

# ---------- RUNTIME ENVs ----------
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV WAN_VAE_PATH=/runpod-volume/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
ENV WORKFLOW_JSON=/runpod-volume/workflows/new_Wan22_api.json

# batching knobs (подбирай под свою карту)
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5

# ---------- ENTRY ----------
CMD ["python3","-u","/app/handler.py"]
