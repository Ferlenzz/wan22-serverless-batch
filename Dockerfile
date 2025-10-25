# ---------- BASE ----------
# Лёгкий рантайм: CUDA 12.1 + cuDNN8 (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- SYSTEM ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ca-certificates \
    git ffmpeg libglib2.0-0 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

# обновим базовые инструменты pip
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# ---------- TORCH (cu121) ----------
# Ставим готовые колёса под CUDA 12.1 — без сборки
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# XFormers (готовое колесо под cu121). FLASH-ATTN НЕ СТАВИМ.
RUN pip install xformers==0.0.27.post2 -f https://download.pytorch.org/whl/cu121

# ---------- WAN2.2 ----------
WORKDIR /app
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

WORKDIR /app/Wan2.2
# Жёстко вырезаем flash-attn/xformers из любых requirements (чтобы не пересобирались)
RUN awk 'BEGIN{IGNORECASE=1} !/flash[-_]?attn/ && !/xformers/ {print}' requirements.txt > /tmp/req-pruned.txt

# Ставим зависимости Wan2.2 из «пропиленного» файла
RUN pip install -r /tmp/req-pruned.txt

# ---------- OUR CODE ----------
WORKDIR /app
# Положи рядом в репозитории эти файлы
COPY handler.py /app/handler.py
COPY engine.py  /app/engine.py

# ---------- RUNPOD SDK + LOGGING ----------
# Важен порядок: сначала совместимые зависимости, потом SDK
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        "pydantic<2" \
        "typing-extensions<4.9" \
        "annotated-types<0.7" && \
    python3 -m pip install --no-cache-dir \
        runpod==1.6.6 \
        loguru

# ---------- RUNTIME ENVs ----------
ENV RP_VOLUME=/runpod-volume
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-TI2V-5B
ENV BATCH_MAX_SIZE=20
ENV BATCH_LINGER_SEC=5
# На всякий случай отключаем любые попытки автосборки flash-attn
ENV FLASH_ATTENTION_SKIP_COMPILE=1
ENV USE_FLASH_ATTENTION=0

# ---------- ENTRY ----------
CMD ["python3","-u","/app/handler.py"]
