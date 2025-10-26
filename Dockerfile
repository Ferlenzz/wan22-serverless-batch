# ---------- BASE ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- SYSTEM ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git curl ffmpeg libglib2.0-0 libgl1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------- TORCH (cu121) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# XFormers wheel (no compile) for cu121
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# ---------- WAN 2.2 code ----------
WORKDIR /app
ARG WAN_REF=main
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2 && \
    cd /app/Wan2.2 && git fetch --all --tags && git checkout ${WAN_REF}

# prune flash-attn/xformers from requirements to avoid compiling
WORKDIR /app/Wan2.2
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt && \
    pip install --no-cache-dir -r /tmp/req-pruned.txt

# ---------- Extra deps we need at runtime ----------
RUN pip install --no-cache-dir \
      runpod==1.6.2 \
      loguru==0.7.2 \
      pillow==10.4.0 \
      imageio==2.36.0 imageio-ffmpeg==0.5.1 \
      numpy==1.26.4 \
      decord==0.6.0 \
      diffusers==0.30.2 \
      # --- PEFT stack ---
      peft==0.17.1 \
      transformers==4.44.2 \
      accelerate==0.34.2 \
      # --- WAN / VAE helpers ---
      einops==0.7.0 \
      safetensors==0.4.5 \
      opencv-python-headless==4.10.0.84 \
      # --- librosa and its deps ---
      librosa==0.10.2.post1 \
      numba==0.60.0 \
      llvmlite==0.43.0 \
      soundfile==0.12.1 \
      audioread==3.0.1

# ---------- APP ----------
WORKDIR /app
COPY engine.py /app/engine.py
COPY handler.py /app/handler.py

# ---------- ENV ----------
ENV RP_VOLUME=/runpod-volume
ENV WAN_ROOT=/app/Wan2.2
ENV PYTHONPATH=/app/Wan2.2:$PYTHONPATH
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
# микробатч сейчас не используется кодом — можно убрать/оставить без влияния
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib

# ---------- ENTRY ----------
CMD ["python3","-u","/app/handler.py"]
